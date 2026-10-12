# WNBA Game Prediction (2023 season)

Can team statistics predict who wins a WNBA game? This project builds a model to predict whether the home team wins, then re-tests it honestly.

## Results at a glance

| Approach (tested on 160 games, late June to September 2023) | Accuracy |
|---|---|
| Model: record + point margin so far | 68.8% |
| Elo rating favorite | 68.8% |
| Team with the better point margin so far | 66.9% |
| Team with the better record so far | 66.2% |
| Model: all 5 pre-game features | 62.5% |
| Always pick the home team | 53.8% |

Every prediction above uses only games played **before** the game being predicted. With 160 test games, differences of a few points are within noise (the full model's 90% interval is 56% to 69%).

## Version 1 vs. Version 2

- **Version 1 (`predict.ipynb`)** reported about 70% accuracy. It used each team's *end-of-season* stats (including total wins and losses) to predict games from the same season, so the stats already contained the results being predicted. Simply picking the team with the better final record scores 73% on the same games.
- **Version 2 (`predict_v2.ipynb`)** rebuilds the features from earlier games only (win percentage, average margin, last-5 form, rest days and an Elo rating) and tests walk-forward through the season.

## 2023 Finals

The Version 2 model favors the Las Vegas Aces over the New York Liberty at either venue. The Aces won the series 3–1, so the model would have called 3 of 4 games.

## What I learned

- Check for data leakage before trusting a high score.
- Always compare against a simple baseline.
- Report uncertainty, especially with one season of data.

**Next steps:** add prior seasons, player availability and injuries, and test on the 2024 season.

## Files

- `reg_season.csv`: 2023 regular-season schedule and scores (240 games)
- `advanced_stats.csv`: 2023 end-of-season team advanced stats (used by Version 1)
- `predict.ipynb`: Version 1
- `predict_v2.ipynb`: Version 2, the honest re-test
- `wnba_model.py`: pre-game features, model and walk-forward test used by the app
- `app.py`: the Streamlit web app
- `requirements.txt`: packages for the app

## Web app

`app.py` is a Streamlit app built on Version 2. Pick a home and away team to get a win probability, see the walk-forward accuracy results, and browse each team's rating.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run the notebook

```bash
pip install pandas scikit-learn jupyter
jupyter notebook predict_v2.ipynb
```

Built with Python, pandas and scikit-learn.
