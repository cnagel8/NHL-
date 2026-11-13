#!/usr/bin/env python3
"""
NHL Shots-on-Goal prop finder  (v2)
===================================
Ranks every skater playing on a given day by their estimated chance of
recording N+ shots on goal (default 2+, i.e. "over 1.5").

What goes into each projection
------------------------------
PLAYER
  * Shots per 60 minutes of ice time (this season, shrunk toward last
    season so early-season samples don't swing wildly), blended with the
    last 10 games for current form.
  * Expected ice time tonight (weighted toward the last 5 games, so
    promotions / demotions in the lineup show up quickly).
OPPONENT DEFENCE (built from every box score of the opponent's games)
  * Shots allowed per game to FORWARDS and to DEFENCEMEN separately, vs.
    league average - a forward is judged by how many shots that defence
    gives up to forwards.  Season and last-10 numbers are blended.
SCHEDULE
  * Back-to-back nights for the player's team (tired legs, fewer shots)
    and for the opponent (tired defence, more shots allowed).
VENUE
  * Small home/road adjustment.

Expected shots -> probability of N+ shots with a Poisson model.  You also
get the player's real hit rates, and an optional comparison to your
sportsbook's odds ("edge").

Before the player table it prints a MATCHUP REPORT for every game: team
shots for/against per game (season and last 10), shots allowed to
forwards and defencemen, and back-to-back flags.

Data: the free public NHL API (api-web.nhle.com, api.nhle.com).
Only the Python standard library is needed.  Results are cached in a
folder called .nhl_cache next to the script, so the first run of the day
is the slow one (a few minutes) and re-runs are fast.

Usage
-----
    python nhl_sog.py                     # today's games, 2+ shots
    python nhl_sog.py --shots 3           # 3+ shots
    python nhl_sog.py --date 2026-11-14   # another day
    python nhl_sog.py --top 40 --csv picks.csv
    python nhl_sog.py --odds odds.csv     # compare to your book's odds
    python nhl_sog.py --no-report         # skip the matchup report

odds.csv format (American odds for the OVER at your --shots line):
    player,odds
    Auston Matthews,-250
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request

WEB = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"

# ---- Model knobs (tweak to taste) -----------------------------------------
PRIOR_MINUTES = 300    # last season's shot rate counts as this many minutes
                       # (~ 17 games for a top-6 forward)
FORM_WEIGHT = 0.30     # share of shot rate taken from the last 10 games
TOI_RECENT = 0.60      # share of expected ice time taken from last 5 games
OPP_WEIGHT = 0.60      # 0 = ignore opponent defence, 1 = full adjustment
OPP_RECENT = 0.35      # share of opponent numbers taken from their last 10
B2B_SELF = 0.97        # player's team played yesterday
B2B_OPP = 1.02         # opponent played yesterday
HOME_BOOST = 1.02
ROAD_PENALTY = 0.98
MIN_SEASON_SOG = 1.3   # skip skaters averaging fewer shots/game
MIN_GAMES = 3
WORKERS = 6            # parallel downloads
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nhl_cache")
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, file=sys.stderr, flush=True)


def get_json(url, cache="day", retries=3):
    """Fetch JSON.  cache='day' keeps it until tomorrow, 'forever' for
    finished-game box scores, None for no cache."""
    path = None
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tag = dt.date.today().isoformat() if cache == "day" else "static"
        key = hashlib.md5(f"{tag}|{url}".encode()).hexdigest()
        path = os.path.join(CACHE_DIR, key + ".json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-sog-script/2.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode("utf-8"))
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch {url}: {e}")
            time.sleep(1.5 * (attempt + 1))


# ---- small helpers ---------------------------------------------------------

def season_id(date):
    y = date.year if date.month >= 8 else date.year - 1
    return f"{y}{y + 1}"


def prev_season_id(sid):
    y = int(sid[:4]) - 1
    return f"{y}{y + 1}"


def name_of(obj):
    return obj.get("default", "") if isinstance(obj, dict) else (obj or "")


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def toi_min(toi):
    try:
        m, s = str(toi).split(":")
        return int(m) + int(s) / 60
    except Exception:
        return None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def poisson_at_least(lam, n):
    if lam <= 0:
        return 0.0
    cdf = sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n))
    return max(0.0, 1.0 - cdf)


def american_to_prob(odds):
    odds = float(odds)
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def prob_to_american(p):
    if p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def is_dman(pos):
    return str(pos).upper().startswith("D")


# ---- schedule --------------------------------------------------------------

def games_on(date):
    data = get_json(f"{WEB}/schedule/{date.isoformat()}")
    for day in data.get("gameWeek", []):
        if day.get("date") == date.isoformat():
            return [g for g in day.get("games", []) if g.get("gameType") in (2, 3)]
    return []


def teams_played_on(date):
    try:
        return {t for g in games_on(date)
                for t in (g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"])}
    except Exception:
        return set()


def completed_game_ids(team, sid, before):
    data = get_json(f"{WEB}/club-schedule-season/{team}/{sid}")
    ids = []
    for g in data.get("games", []):
        if (g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL")
                and str(g.get("gameDate", ""))[:10] < before.isoformat()):
            ids.append((str(g.get("gameDate", ""))[:10], g["id"]))
    ids.sort()
    return [i for _, i in ids]


# ---- box scores -> team shot profiles ---------------------------------------

def parse_boxscore(box):
    """Return {abbrev: {'F': shots, 'D': shots}} for both teams in a game."""
    out = {}
    pbg = box.get("playerByGameStats", {})
    for side in ("homeTeam", "awayTeam"):
        abbr = box.get(side, {}).get("abbrev")
        grp = pbg.get(side, {})
        f = sum((p.get("sog", p.get("shots", 0)) or 0) for p in grp.get("forwards", []))
        d = sum((p.get("sog", p.get("shots", 0)) or 0)
                for p in grp.get("defense", grp.get("defensemen", [])))
        if abbr:
            out[abbr] = {"F": f, "D": d}
    return out


def build_team_profiles(teams, sid, date):
    """For each team: per-game list of shots FOR and AGAINST split F/D."""
    log("Collecting box scores for team defence stats ...")
    game_ids = set()
    per_team_ids = {}
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(completed_game_ids, t, sid, date): t for t in teams}
        for fu in cf.as_completed(futs):
            t = futs[fu]
            try:
                per_team_ids[t] = fu.result()
                game_ids.update(per_team_ids[t])
            except Exception as e:
                log(f"  no schedule for {t}: {e}")
                per_team_ids[t] = []

    boxes = {}
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(get_json, f"{WEB}/gamecenter/{gid}/boxscore", "forever"): gid
                for gid in game_ids}
        for fu in cf.as_completed(futs):
            try:
                boxes[futs[fu]] = parse_boxscore(fu.result())
            except Exception as e:
                log(f"  skipped box score {futs[fu]}: {e}")

    profiles = {}
    league = {"F": [], "D": [], "T": []}
    for t in teams:
        rows = []
        for gid in per_team_ids.get(t, []):
            b = boxes.get(gid)
            if not b or t not in b or len(b) != 2:
                continue
            opp = next(k for k in b if k != t)
            rows.append({"for_F": b[t]["F"], "for_D": b[t]["D"],
                         "ag_F": b[opp]["F"], "ag_D": b[opp]["D"]})
        profiles[t] = rows
    # league averages from every box score we saw (both sides of each game)
    for b in boxes.values():
        for side in b.values():
            league["F"].append(side["F"])
            league["D"].append(side["D"])
            league["T"].append(side["F"] + side["D"])
    lg = {k: avg(v) for k, v in league.items()}
    return profiles, lg


def summarize(rows, key, last=None):
    rows = rows[-last:] if last else rows
    return avg([r[key] for r in rows])


def team_line(rows):
    """Season and last-10 shots for/against (total, F, D)."""
    s = {}
    for k in ("for_F", "for_D", "ag_F", "ag_D"):
        s[k] = summarize(rows, k)
        s[k + "_10"] = summarize(rows, k, 10)
    for scope in ("", "_10"):
        s["for" + scope] = (s["for_F" + scope] or 0) + (s["for_D" + scope] or 0)
        s["ag" + scope] = (s["ag_F" + scope] or 0) + (s["ag_D" + scope] or 0)
    s["gp"] = len(rows)
    return s


def opp_factor(opp_rows, pos_key, lg_avg):
    """How generous this defence is to forwards ('F') or D-men ('D')."""
    if not opp_rows or not lg_avg:
        return 1.0
    season = summarize(opp_rows, "ag_" + pos_key)
    recent = summarize(opp_rows, "ag_" + pos_key, 10)
    allowed = (1 - OPP_RECENT) * season + OPP_RECENT * recent
    # shrink small samples toward league average (8 game prior)
    n = len(opp_rows)
    allowed = (allowed * n + lg_avg * 8) / (n + 8)
    return 1 + OPP_WEIGHT * (allowed / lg_avg - 1)


# ---- players ---------------------------------------------------------------

def team_skaters(team, sid):
    return get_json(f"{WEB}/club-stats/{team}/{sid}/2").get("skaters", [])


def player_game_log(pid, sid):
    return get_json(f"{WEB}/player/{pid}/game-log/{sid}/2").get("gameLog", [])


def prev_season(pid, prev_sid):
    """(shots, games, minutes) for last NHL regular season."""
    try:
        data = get_json(f"{WEB}/player/{pid}/landing")
    except Exception:
        return 0, 0, 0
    shots = gp = mins = 0
    for s in data.get("seasonTotals", []):
        if (str(s.get("season")) == prev_sid and s.get("gameTypeId") == 2
                and s.get("leagueAbbrev") == "NHL"):
            g = s.get("gamesPlayed", 0) or 0
            shots += s.get("shots", 0) or 0
            gp += g
            t = toi_min(s.get("avgToi", ""))
            mins += (t or 0) * g
    return shots, gp, mins


def project(glog, prev, factor, n):
    shots = [g.get("shots", 0) or 0 for g in glog]       # newest first
    tois = [toi_min(g.get("toi")) for g in glog]
    gp = len(shots)
    pairs = [(s, t) for s, t in zip(shots, tois) if t]
    cur_mins = sum(t for _, t in pairs)
    cur_shots = sum(s for s, _ in pairs)
    season_toi = avg(tois) or 15.0

    # shot rate per minute, shrunk toward last season
    p_shots, p_gp, p_mins = prev
    if p_gp >= 10:
        if not p_mins:                       # no TOI in landing data
            p_mins = p_gp * season_toi
        prior_rate = p_shots / p_mins
        rate = (cur_shots + PRIOR_MINUTES * prior_rate) / (cur_mins + PRIOR_MINUTES)
    else:
        prior_rate = None
        rate = cur_shots / cur_mins if cur_mins else sum(shots) / gp / season_toi

    l10 = [(s, t) for s, t in pairs[:10]]
    if l10:
        l10_rate = sum(s for s, _ in l10) / sum(t for _, t in l10)
        rate = (1 - FORM_WEIGHT) * rate + FORM_WEIGHT * l10_rate

    exp_toi = TOI_RECENT * (avg(tois[:5]) or season_toi) + (1 - TOI_RECENT) * season_toi
    lam = rate * exp_toi * factor

    last10 = shots[:10]
    return {
        "gp": gp,
        "season_avg": sum(shots) / gp,
        "last10_avg": sum(last10) / len(last10),
        "prev_avg": p_shots / p_gp if p_gp else None,
        "sog_per60": rate * 60,
        "toi_season": season_toi,
        "toi_exp": exp_toi,
        "factor": factor,
        "exp_sog": lam,
        "model_p": poisson_at_least(lam, n),
        "hit_season": sum(s >= n for s in shots) / gp,
        "hit_last10": sum(s >= n for s in last10) / len(last10),
        "last5": shots[:5],
    }


def load_odds(path):
    odds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                odds[norm(row["player"])] = float(row["odds"])
            except (KeyError, ValueError):
                continue
    return odds


# ---- output ----------------------------------------------------------------

def fmt(x, d=1):
    return "  -" if x is None else f"{x:.{d}f}"


def print_report(games, profiles, lg, b2b):
    print("\nMATCHUP REPORT  (shots on goal per game; L10 = last 10 games)")
    print(f"League avg per team: {fmt(lg['T'])} total  |  "
          f"{fmt(lg['F'])} by forwards  |  {fmt(lg['D'])} by defencemen\n")
    hdr = (f"  {'Team':<5}{'GP':>3}  {'For':>5} {'L10':>5}   {'Agst':>5} {'L10':>5}"
           f"   {'Ag-F':>5} {'Ag-D':>5}  B2B")
    for g in games:
        a, h = g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]
        print(f"{a} @ {h}")
        print(hdr)
        for t in (a, h):
            s = team_line(profiles.get(t, []))
            print(f"  {t:<5}{s['gp']:>3}  {fmt(s['for']):>5} {fmt(s['for_10']):>5}   "
                  f"{fmt(s['ag']):>5} {fmt(s['ag_10']):>5}   "
                  f"{fmt(s['ag_F']):>5} {fmt(s['ag_D']):>5}  "
                  f"{'YES' if t in b2b else ''}")
        print()
    print("For/Agst = shots taken/allowed.  Ag-F / Ag-D = shots allowed to "
          "forwards / defencemen.\nHigher Agst means a juicier matchup for "
          "the other team's shooters.")


def print_players(results, n, date, odds, top):
    print(f"\nNHL {n}+ SHOTS ON GOAL  |  {date}  |  ranked by "
          f"{'edge vs book' if odds else 'model probability'}")
    hdr = (f"{'#':>2} {'Player':<21}{'Pos':<4}{'vs':<5}{'GP':>3} {'Szn':>4} {'L10':>4}"
           f" {'/60':>4} {'TOI':>5} {'Opp':>5} {'Exp':>4} {'Model':>6} {'Fair':>6}"
           f" {'Hit%':>5} {'L10%':>5}  {'Last 5':<12}")
    if odds:
        hdr += f"{'Book':>6} {'Edge':>6}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results[:top], 1):
        fair = prob_to_american(r["model_p"])
        vs = ("" if r["venue"] == "H" else "@") + r["opp"]
        flag = "*" if r["b2b"] else " "
        line = (f"{i:>2} {r['player'][:19] + flag:<21}{r['pos']:<4}{vs:<5}{r['gp']:>3}"
                f" {r['season_avg']:>4.1f} {r['last10_avg']:>4.1f} {r['sog_per60']:>4.1f}"
                f" {r['toi_exp']:>5.1f} {(r['factor'] - 1) * 100:>+4.0f}%"
                f" {r['exp_sog']:>4.1f} {r['model_p'] * 100:>5.0f}%"
                f" {(f'{fair:+d}' if fair else '-'):>6}"
                f" {r['hit_season'] * 100:>4.0f}% {r['hit_last10'] * 100:>4.0f}%"
                f"  {'-'.join(map(str, r['last5'])):<12}")
        if odds:
            line += (f"{r['book_odds']:>+6.0f} {r['edge'] * 100:>+5.1f}%"
                     if "book_odds" in r else f"{'-':>6} {'-':>6}")
        print(line)
    print("""
Szn/L10 = shots per game (season / last 10).  /60 = shots per 60 min of ice
time.  TOI = expected minutes tonight.  Opp = total matchup boost or penalty
(opponent defence vs his position, back-to-backs, home/road).  Exp = projected
shots.  Model = chance of hitting.  Fair = odds that chance is worth; only bet
if your book pays better.  * = his team played yesterday.
Check confirmed lineups - this can't see scratches or injuries.""")


# ---- main ------------------------------------------------------------------

def analyze(date, n=2, min_sog=MIN_SEASON_SOG, odds=None):
    """Do all the work for one day.  Returns a dict with games, team
    profiles, league averages, back-to-back teams and player results.
    Raises RuntimeError if the NHL API can't be reached."""
    odds = odds or {}
    sid, psid = season_id(date), prev_season_id(season_id(date))
    log(f"Loading NHL games for {date} ...")
    games = games_on(date)
    out = {"games": games, "profiles": {}, "lg": {}, "b2b": set(), "results": []}
    if not games:
        return out

    matchups = []
    for g in games:
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        matchups += [(h, a, True), (a, h, False)]
    teams = sorted({m[0] for m in matchups})
    log("  " + ", ".join(f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}"
                         for g in games))

    b2b = teams_played_on(date - dt.timedelta(days=1)) & set(teams)
    profiles, lg = build_team_profiles(teams, sid, date)

    candidates = []
    for team, opp, is_home in matchups:
        try:
            skaters = team_skaters(team, sid)
        except Exception as e:
            log(f"  skipped {team}: {e}")
            continue
        for p in skaters:
            gp = p.get("gamesPlayed", 0) or 0
            if gp >= MIN_GAMES and (p.get("shots", 0) or 0) / gp >= min_sog:
                candidates.append((p, team, opp, is_home))
    log(f"Analyzing {len(candidates)} players ...")

    def work(c):
        p, team, opp, is_home = c
        pid = p["playerId"]
        glog = player_game_log(pid, sid)
        if len(glog) < MIN_GAMES:
            return None
        pos = p.get("positionCode", "")
        factor = opp_factor(profiles.get(opp), "D" if is_dman(pos) else "F",
                            lg["D" if is_dman(pos) else "F"])
        if team in b2b:
            factor *= B2B_SELF
        if opp in b2b:
            factor *= B2B_OPP
        factor *= HOME_BOOST if is_home else ROAD_PENALTY
        r = project(glog, prev_season(pid, psid), factor, n)
        r["shots_log"] = [g.get("shots", 0) or 0 for g in glog]
        name = f"{name_of(p.get('firstName'))} {name_of(p.get('lastName'))}"
        r.update({"player": name, "team": team, "opp": opp, "pos": pos,
                  "venue": "H" if is_home else "A", "b2b": team in b2b})
        book = odds.get(norm(name))
        if book is not None:
            r.update(book_odds=book, book_p=american_to_prob(book),
                     edge=r["model_p"] - american_to_prob(book))
        return r

    results = []
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for fu in cf.as_completed([ex.submit(work, c) for c in candidates]):
            try:
                r = fu.result()
                if r:
                    results.append(r)
            except Exception as e:
                log(f"  skipped a player: {e}")
    out.update(profiles=profiles, lg=lg, b2b=b2b, results=results)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rank NHL shots-on-goal props.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--shots", type=int, default=2, help="2 means 2+ (over 1.5)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--csv", help="save full results to this CSV")
    ap.add_argument("--odds", help="CSV with player,odds (American)")
    ap.add_argument("--min-sog", type=float, default=MIN_SEASON_SOG)
    ap.add_argument("--no-report", action="store_true", help="skip matchup report")
    args = ap.parse_args()

    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    n = args.shots
    odds = load_odds(args.odds) if args.odds else {}
    try:
        res = analyze(date, n, args.min_sog, odds)
    except RuntimeError as e:
        print(f"Couldn't reach the NHL API: {e}")
        return
    games, profiles, lg, b2b, results = (res["games"], res["profiles"], res["lg"],
                                         res["b2b"], res["results"])
    if not games:
        print(f"No NHL games found on {date}.")
        return

    if not args.no_report:
        print_report(games, profiles, lg, b2b)
    if not results:
        print("No qualifying players found.")
        return
    key = "edge" if odds else "model_p"
    results.sort(key=lambda r: r.get(key, -1), reverse=True)
    print_players(results, n, date, odds, args.top)

    if args.csv:
        fields = ["player", "team", "pos", "opp", "venue", "b2b", "gp",
                  "season_avg", "last10_avg", "prev_avg", "sog_per60",
                  "toi_season", "toi_exp", "factor", "exp_sog", "model_p",
                  "hit_season", "hit_last10", "book_odds", "book_p", "edge"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nSaved {len(results)} players to {args.csv}")


if __name__ == "__main__":
    main()
