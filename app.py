import pandas as pd
import streamlit as st

import wnba_model as wm

st.set_page_config(page_title="WNBA Matchup Predictor", page_icon="🏀", layout="centered")


@st.cache_data
def load():
    games = wm.load_games("reg_season.csv")
    F, teams = wm.build_features(games)
    return games, F, teams


@st.cache_resource
def model(_F):
    return wm.train_final_model(_F)


@st.cache_data
def accuracy(_F):
    return wm.walk_forward(_F)


games, F, teams = load()
clf = model(F)
names = sorted(teams)

st.title("WNBA Matchup Predictor")
st.caption("2023 season · built from regular-season results only, using information available before each game")

tab_pred, tab_acc, tab_teams = st.tabs(["Predict a game", "How accurate is it?", "Team ratings"])

with tab_pred:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", names, index=names.index("Las Vegas Aces"))
    away_options = [n for n in names if n != home]
    default_away = "New York Liberty" if "New York Liberty" in away_options else away_options[0]
    away = c2.selectbox("Away team", away_options, index=away_options.index(default_away))

    p_model, p_elo = wm.predict(clf, teams, home, away)
    winner, p_win = (home, p_model) if p_model >= 0.5 else (away, 1 - p_model)

    st.subheader(f"Pick: {winner}")
    st.progress(p_model, text=f"{home} win probability: {p_model:.0%}  ·  {away}: {1 - p_model:.0%}")
    m1, m2 = st.columns(2)
    m1.metric("Model confidence", f"{p_win:.0%}")
    m2.metric(f"Elo rating says ({home})", f"{p_elo:.0%}")

    t = wm.team_table(teams).set_index("Team").loc[[home, away]]
    st.dataframe(t, width="stretch")
    st.caption("Ratings reflect the full 2023 regular season. The model uses each team's win percentage and average point margin.")

with tab_acc:
    with st.spinner("Re-running the walk-forward test…"):
        acc, n = accuracy(F)
    st.write(f"Each of the last **{n} games** of the season was predicted by a model trained only on games played before it.")
    st.dataframe(acc.map("{:.1%}".format).rename("Accuracy"), width="stretch")
    st.info("With only 160 test games, a few points of difference is within noise. "
            "Every honest approach lands between about 63% and 69%, and a simple Elo rating does as well as the model.")
    with st.expander("Why not the original 70%?"):
        st.write("The first version of this project used end-of-season statistics, including each team's total losses, "
                 "to predict games from the same season. Those totals already contained the results being predicted. "
                 "This app rebuilds every feature from earlier games only. Full details are in `predict_v2.ipynb`.")

with tab_teams:
    st.dataframe(wm.team_table(teams), width="stretch", hide_index=True)
    st.caption("Elo starts every team at 1500 and moves after each game based on the result, the margin and how surprising it was.")
