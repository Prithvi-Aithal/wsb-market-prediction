# Predicting market volatility & short-term trading signals from r/WallStreetBets

UE24CS352A – Machine Learning mini-project. Re-implementation and extension of the CS229 report
*"Predicting market volatility and building short-term trading strategies using data from Reddit's WallStreetBets"*.

## What it does
1. **Data** – Kaggle `gpreda/reddit-wallstreetsbets-posts` (53,187 posts, Sep 2020 – Aug 2021) + daily prices from Yahoo Finance (`yfinance`).
2. **Preprocessing** – drop NaN titles, lowercase, strip URLs/handles/special/single characters, collapse spaces; filter posts that mention GME, AMC, TSLA or FB.
3. **Features (7 per post)** – post_score, total_comms_num, len_of_post, positive/neutral/negative sentiment (VADER), bias column.
4. **Labels** – `direction` (next open > this open) and `volatility` (|next-open move| > training-set median, i.e. a "significant shift"). Two timing conventions: `paper` (as in the report) and `tradable` (only the move after the post could be traded – no look-ahead).
5. **Models** – logistic regression solved with **Newton's method written from scratch**; 3×256 ReLU network with sigmoid head (Adam, lr 1e-3, 30 epochs); uniform-random and majority-class baselines.
6. **Beyond replication – tuning** – L2 strength of the logistic regression and a random search over the network (depth/width, learning rate, weight decay, batch size) chosen **on the validation split only**; both "report config" and "tuned" results are reported.
7. **Strategy back-test** – per day, majority vote of per-post predictions → long / flat, compared with always-long.

Splits are **chronological 80/10/10** (train/val/test), so the test set is strictly in the future.

## Setup
Python 3.10+.
```bash
pip install -r requirements.txt
```
Download the dataset (no Kaggle login needed for this public dataset) into `data/`:
```bash
python -c "import kagglehub,shutil,glob; p=kagglehub.dataset_download('gpreda/reddit-wallstreetsbets-posts'); shutil.copy(glob.glob(p+'/*.csv')[0],'data/reddit_wsb.csv')"
```
Prices are downloaded and cached automatically on first run.

## Run
```bash
python src/run_experiments.py          # all experiments -> results/ (tables, figures, backtest) and models/
python src/demo.py --stock GME --title "GME to the moon, diamond hands" --score 1200 --comments 340
python src/demo.py --stock TSLA --task volatility --title "Tesla earnings tomorrow"
python src/demo.py --replay GME        # back-test summary for a stock
python docs/build_docs.py              # regenerates the 2-page write-up PDF and the slide deck
```
`run_experiments.py` takes several minutes (neural-net seeds + hyper-parameter search).

## Layout
```
src/wsb_pipeline.py     data loading, preprocessing, features, labels, splits
src/models.py           Newton logistic regression, neural network
src/run_experiments.py  experiments, tuning, baselines, back-test, figures
src/demo.py             live demo
results/                results.csv/json, backtest.csv, figures
docs/                   write-up PDF + presentation
```

## Caveats
Posts are a noisy, small-sample signal; test sets for TSLA/FB are only ~25–125 posts (standard error 5–10 pp), so
differences of a few points are not significant. Not financial advice.
