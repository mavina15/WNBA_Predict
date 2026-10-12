import pandas as pd
import streamlit as st

import wnba_model as wm

st.set_page_config(page_title="WNBA Prediction Web App", page_icon="🏀", layout="centered")

# Team colors carried over from the original 2023 Flask app
TEAM_COLORS = {
    "Los Angeles Sparks": "#1D1160", "Las Vegas Aces": "#C8102E", "Connecticut Sun": "#F05D2F",
    "Phoenix Mercury": "#E56020", "Chicago Sky": "#5FA1D7", "Indiana Fever": "#FFBF00",
    "Washington Mystics": "#07294D", "New York Liberty": "#003B8B", "Atlanta Dream": "#E03A3E",
    "Seattle Storm": "#00586D", "Dallas Wings": "#00529B", "Minnesota Lynx": "#0C2340",
}
LIGHT_TEXT = {"Indiana Fever", "Chicago Sky"}   # light team colors need dark text

st.markdown("""
<style>
h1 {font-weight: 800; letter-spacing: -0.02em;}
[data-testid="stForm"] {border: 3px solid #F2A33A !important; border-radius: 16px; padding: 1.25rem 1.5rem;}
[data-testid="stFormSubmitButton"] button {background: #5CB85C; color: #fff; border: none; font-weight: 700; padding: 0.5rem 1.6rem;}
[data-testid="stFormSubmitButton"] button:hover {background: #4CA34C; color: #fff;}
.team-card {border-radius: 14px; padding: 1rem 1.1rem; color: #fff; height: 100%;}
.team-card h4 {margin: 0 0 .15rem 0; color: inherit; font-size: 1.05rem;}
.team-card .role {font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; opacity: .85;}
.team-card .prob {font-size: 2.3rem; font-weight: 800; line-height: 1.1; margin: .35rem 0 .5rem;}
.team-card table {width: 100%; font-size: .85rem; border-collapse: collapse;}
.team-card table, .team-card tr, .team-card td {border: none !important; background: transparent !important;}
.team-card td {padding: .18rem 0; border-top: 1px solid rgba(255,255,255,.25) !important;}
.team-card td:last-child {text-align: right; font-variant-numeric: tabular-nums;}
.team-card.dark-text {color: #1F2937;}
.team-card.dark-text td {border-top-color: rgba(0,0,0,.15) !important;}
.winner {display: inline-block; background: #5CB85C; color: #fff; font-weight: 700; padding: .35rem .9rem; border-radius: 999px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    games = wm.load_games("reg_season.csv")
    F, teams = wm.build_features(games)
    adv = pd.read_csv("advanced_stats.csv").dropna(axis=1, how="all")
    adv = adv[adv["Team"] != "League Average"].set_index("Team")[["ORtg", "DRtg", "NRtg", "Pace"]]
    return games, F, teams, adv


@st.cache_resource
def model(_F):
    return wm.train_final_model(_F)


@st.cache_data
def accuracy(_F):
    return wm.walk_forward(_F)


games, F, teams, adv = load()
clf = model(F)
names = sorted(teams)
ratings = wm.team_table(teams).set_index("Team")


def team_card(team, role, prob):
    r, a = ratings.loc[team], adv.loc[team]
    cls = "team-card dark-text" if team in LIGHT_TEXT else "team-card"
    rows = [("Record", f"{int(r.W)}–{int(r.L)}"), ("Avg point margin", f"{r['Avg margin']:+.1f}"),
            ("Last 5 games", f"{r['Last 5 margin']:+.1f}"), ("Elo rating", f"{int(r.Elo)}"),
            ("Offensive rating", f"{a.ORtg:.1f}"), ("Defensive rating", f"{a.DRtg:.1f}")]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f'<div class="{cls}" style="background:{TEAM_COLORS.get(team, "#374151")}">'
            f'<div class="role">{role}</div><h4>{team}</h4><div class="prob">{prob:.0%}</div>'
            f'<table>{body}</table></div>')


st.title("🏀 WNBA Prediction Web App")
st.caption("2023 season · predictions use only information available before each game")

tab_pred, tab_acc, tab_teams, tab_about = st.tabs(["Predict", "How accurate is it?", "Team ratings", "About"])

with tab_pred:
    with st.form("matchup"):
        st.markdown("**Select teams:**")
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home team", names, index=names.index("Las Vegas Aces"))
        away = c2.selectbox("Away team", names, index=names.index("New York Liberty"))
        submitted = st.form_submit_button("Predict")

    if home == away:
        st.warning("Please choose two different teams.")
    else:
        p_model, p_elo = wm.predict(clf, teams, home, away)
        winner = home if p_model >= 0.5 else away
        st.markdown(f'<span class="winner">Predicted winner: {winner}</span>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.markdown(team_card(home, "Home", p_model), unsafe_allow_html=True)
        c2.markdown(team_card(away, "Away", 1 - p_model), unsafe_allow_html=True)
        st.caption(f"The model uses each team's win percentage and average point margin. "
                   f"For comparison, the Elo rating gives {home} a {p_elo:.0%} chance at home. "
                   f"Offensive and defensive ratings are points per 100 possessions, shown for context.")

with tab_acc:
    with st.spinner("Re-running the walk-forward test…"):
        acc, n = accuracy(F)
    st.write(f"Each of the last **{n} games** of the season was predicted by a model trained only on games played before it.")
    st.dataframe(acc.map("{:.1%}".format).rename("Accuracy"), width="stretch")
    st.info("With only 160 test games, a few points of difference is within noise. "
            "Every honest approach lands between about 63% and 69%, and a simple Elo rating does as well as the model.")

with tab_teams:
    t = ratings.join(adv).reset_index()
    st.dataframe(t, width="stretch", hide_index=True)
    st.caption("Elo starts every team at 1500 and moves after each game based on the result, the margin and how surprising it was.")

with tab_about:
    st.subheader("About this project")
    st.write("I first built this app in 2023 with Flask. That version predicted winners from each team's end-of-season "
             "statistics and reported about 70% accuracy. When I re-tested it, I found those statistics already contained "
             "the results being predicted, so the score was inflated.")
    st.write("This version rebuilds every feature from earlier games only: win percentage, point margin, recent form, "
             "rest and an Elo rating. Tested game by game through the season, it lands at about 69%, tied with a simple Elo rating.")
    st.write("Code, data and both notebooks: [github.com/mavina15/WNBA_Predict](https://github.com/mavina15/WNBA_Predict)  \n"
             "Case study: [mavinabeltran.com](https://mavinabeltran.com/wnba-prediction.html)")
