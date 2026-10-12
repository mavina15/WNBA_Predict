"""Pre-game features, a simple model and a walk-forward test for the 2023 WNBA season.

Every feature for a game is built only from games played before it, so nothing
about the game being predicted can leak into its prediction.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K, HOME_EDGE = 20, 50          # Elo settings
MODEL_FEATURES = ["d_wp", "d_mov"]
ALL_FEATURES = ["d_wp", "d_mov", "d_l5", "d_rest", "d_elo"]
MIN_GAMES = 5                  # both teams need this many games before a game is used for training
TEST_START = 80                # games 81-240 are the walk-forward test set


def load_games(path="reg_season.csv"):
    s = pd.read_csv(path).rename(columns={"Visitor/Neutral": "away", "PTS": "away_pts",
                                          "Home/Neutral": "home", "PTS.1": "home_pts"})
    s["date"] = pd.to_datetime(s["Date"], format="%a %b %d %Y")
    s = s.sort_values("date", kind="stable").reset_index(drop=True)
    s["home_win"] = (s.home_pts > s.away_pts).astype(int)
    return s[["date", "away", "away_pts", "home", "home_pts", "home_win"]]


def _win_pct(r):
    return (r["w"] + 1) / (r["w"] + r["l"] + 2)          # lightly smoothed


def _avg_margin(r, last=None):
    m = r["margins"][-last:] if last else r["margins"]
    return float(np.mean(m)) if m else 0.0


def build_features(games):
    """Return (features per game, team state after the last game)."""
    teams = {t: dict(w=0, l=0, margins=[], last=None, elo=1500.0) for t in set(games.home) | set(games.away)}
    rows = []
    for i, g in games.iterrows():
        h, a = teams[g.home], teams[g.away]
        rest = lambda r: min((g.date - r["last"]).days, 7) if r["last"] is not None else 7
        elo_p = 1 / (1 + 10 ** (-(h["elo"] + HOME_EDGE - a["elo"]) / 400))
        rows.append(dict(i=i, date=g.date, home=g.home, away=g.away, y=g.home_win,
                         gp_min=min(h["w"] + h["l"], a["w"] + a["l"]),
                         d_wp=_win_pct(h) - _win_pct(a), d_mov=_avg_margin(h) - _avg_margin(a),
                         d_l5=_avg_margin(h, 5) - _avg_margin(a, 5), d_rest=rest(h) - rest(a),
                         d_elo=h["elo"] - a["elo"], elo_p=elo_p))
        # update AFTER recording this game's pre-game features
        m = g.home_pts - g.away_pts
        h["margins"].append(m); a["margins"].append(-m)
        if m > 0: h["w"] += 1; a["l"] += 1
        else: a["w"] += 1; h["l"] += 1
        h["last"] = a["last"] = g.date
        d = K * np.log(abs(m) + 1) * ((m > 0) - elo_p)
        h["elo"] += d; a["elo"] -= d
    return pd.DataFrame(rows), teams


def _fit(train, feats):
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(train[feats], train.y)


def train_final_model(F, feats=MODEL_FEATURES):
    return _fit(F[F.gp_min >= MIN_GAMES], feats)


def team_table(teams):
    rows = [dict(Team=t, W=r["w"], L=r["l"], **{"Avg margin": round(_avg_margin(r), 1),
                 "Last 5 margin": round(_avg_margin(r, 5), 1), "Elo": round(r["elo"])}) for t, r in teams.items()]
    return pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)


def predict(model, teams, home, away, feats=MODEL_FEATURES):
    h, a = teams[home], teams[away]
    x = pd.DataFrame([dict(d_wp=_win_pct(h) - _win_pct(a), d_mov=_avg_margin(h) - _avg_margin(a),
                           d_l5=_avg_margin(h, 5) - _avg_margin(a, 5), d_rest=0, d_elo=h["elo"] - a["elo"])])
    p_model = float(model.predict_proba(x[feats])[0, 1])
    p_elo = 1 / (1 + 10 ** (-(h["elo"] + HOME_EDGE - a["elo"]) / 400))
    return p_model, float(p_elo)


def walk_forward(F):
    """Predict each test game from games played before its date only. Returns accuracy by approach."""
    ev = F[F.i >= TEST_START].copy()
    out = {}
    for name, feats in (("Model: record + margin so far", MODEL_FEATURES), ("Model: all 5 pre-game features", ALL_FEATURES)):
        preds = []
        for _, g in ev.iterrows():
            m = _fit(F[(F.date < g.date) & (F.gp_min >= MIN_GAMES)], feats)
            preds.append(int(m.predict_proba(pd.DataFrame([g[feats]]))[0, 1] >= 0.5))
        out[name] = float(np.mean(np.array(preds) == ev.y.values))
    out["Elo rating favorite"] = float(np.mean((ev.elo_p >= 0.5).astype(int) == ev.y))
    out["Better point margin so far"] = float(np.mean((ev.d_mov > 0).astype(int) == ev.y))
    out["Better record so far"] = float(np.mean((ev.d_wp > 0).astype(int) == ev.y))
    out["Always pick the home team"] = float(ev.y.mean())
    return pd.Series(out).sort_values(ascending=False), len(ev)
