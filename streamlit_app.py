"""
NHL Shots-on-Goal picks - web version.
Runs the same model as nhl_sog.py and shows it as a phone-friendly page.
Host free on Streamlit Community Cloud (see setup steps).
"""

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import nhl_sog as m

st.set_page_config(page_title="NHL SOG Picks", page_icon="🏒", layout="wide")
st.title("🏒 NHL Shots on Goal Picks")

today_et = dt.datetime.now(ZoneInfo("America/New_York")).date()

with st.sidebar:
    st.header("Settings")
    date = st.date_input("Game date", today_et)
    min_sog = st.slider("Only players averaging at least (shots/game)",
                        0.5, 3.0, 1.0, 0.1)
    st.markdown("**Your sportsbook odds (optional)**")
    odds_line = st.selectbox("These odds are for", [1, 2, 3, 4], index=1,
                             format_func=lambda k: f"{k}+ shots")
    odds_text = st.text_area("One per line: Player Name, odds",
                             placeholder="Auston Matthews, -250\nCole Caufield, -180")
    st.caption("Can't see injuries or scratches - check lineups before betting.")


@st.cache_data(ttl=1800, show_spinner=False)
def load(date_str, min_sog):
    res = m.analyze(dt.date.fromisoformat(date_str), 2, min_sog)
    res["b2b"] = sorted(res["b2b"])
    return res


def parse_odds(text):
    odds = {}
    for line in text.splitlines():
        if "," not in line:
            continue
        name, o = line.rsplit(",", 1)
        try:
            odds[m.norm(name.strip())] = float(o.strip().replace("+", ""))
        except ValueError:
            pass
    return odds


with st.spinner("Crunching the numbers... first load of the day can take a few minutes."):
    try:
        res = load(date.isoformat(), min_sog)
    except RuntimeError as e:
        st.error(f"Couldn't reach the NHL stats site. Try again in a minute.\n\n{e}")
        st.stop()

games, results = res["games"], res["results"]
if not games:
    st.info(f"No NHL games on {date:%A %b %d}.")
    st.stop()
if not results:
    st.warning("Games found, but no player data yet.")
    st.stop()

odds = parse_odds(odds_text)
b2b = set(res["b2b"])

labels = [f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}" for g in games]
pick_game = st.selectbox("Game", ["All games"] + labels)


def table(n):
    rows = []
    for r in results:
        if pick_game != "All games" and pick_game not in (
                f"{r['opp']} @ {r['team']}", f"{r['team']} @ {r['opp']}"):
            continue
        log = r["shots_log"]
        p = m.poisson_at_least(r["exp_sog"], n)
        fair = m.prob_to_american(p)
        row = {
            "Player": r["player"] + (" *" if r["b2b"] else ""),
            "Team": r["team"],
            "vs": ("" if r["venue"] == "H" else "@") + r["opp"],
            "Pos": r["pos"],
            "Exp shots": round(r["exp_sog"], 1),
            "Chance": round(p * 100),
            "Fair odds": f"{fair:+d}" if fair else "-",
            "Hit % season": round(100 * sum(s >= n for s in log) / len(log)),
            "Hit % L10": round(100 * sum(s >= n for s in log[:10]) / len(log[:10])),
            "Last 5": "-".join(map(str, log[:5])),
            "Shots/gm": round(r["season_avg"], 1),
            "L10 shots/gm": round(r["last10_avg"], 1),
            "Shots/60": round(r["sog_per60"], 1),
            "TOI tonight": round(r["toi_exp"], 1),
            "Matchup": f"{(r['factor'] - 1) * 100:+.0f}%",
        }
        book = odds.get(m.norm(r["player"])) if n == odds_line else None
        if book is not None:
            bp = m.american_to_prob(book)
            row["Book"] = f"{book:+.0f}"
            row["Edge"] = round((p - bp) * 100, 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort = "Edge" if "Edge" in df and df["Edge"].notna().any() else "Chance"
    return df.sort_values(sort, ascending=False, na_position="last").reset_index(drop=True)


tab2, tab1, tab3, tabm = st.tabs(["2+ shots", "1+ shots", "3+ shots", "Matchups"])
for tab, n in ((tab2, 2), (tab1, 1), (tab3, 3)):
    with tab:
        df = table(n)
        if df.empty:
            st.write("No players for this game.")
            continue
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={
                "Chance": st.column_config.ProgressColumn(
                    "Chance", format="%d%%", min_value=0, max_value=100),
                "Edge": st.column_config.NumberColumn("Edge", format="%+.1f%%"),
            })

with tabm:
    lg = res["lg"]
    st.caption(f"League average per team per game: {m.fmt(lg.get('T'))} shots "
               f"({m.fmt(lg.get('F'))} by forwards, {m.fmt(lg.get('D'))} by D-men).")
    rows = []
    for g in games:
        for t in (g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]):
            s = m.team_line(res["profiles"].get(t, []))
            rows.append({
                "Game": f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}",
                "Team": t, "GP": s["gp"],
                "Shots for": round(s["for"], 1), "For L10": round(s["for_10"], 1),
                "Shots against": round(s["ag"], 1), "Agst L10": round(s["ag_10"], 1),
                "Allowed to F": round(s["ag_F"] or 0, 1),
                "Allowed to D": round(s["ag_D"] or 0, 1),
                "Back-to-back": "YES" if t in b2b else "",
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("How to read this"):
    st.markdown("""
- **Exp shots**: projected shots tonight. **Chance**: model's chance he hits the line.
- **Fair odds**: what that chance is worth. Only bet if your book pays *better*
  (e.g. Fair -200, book -150 = value; book -300 = no value).
- **Hit %**: how often he actually hit it (season / last 10).
- **Matchup**: boost or penalty from the opponent's defence vs his position,
  back-to-backs and home ice. **\\*** = his team played yesterday.
- Paste your book's odds in the sidebar to get an **Edge** column (model % minus book %).
""")
