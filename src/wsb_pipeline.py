"""Data loading, preprocessing, feature extraction and label construction.

Mirrors the pipeline in the CS229 report: Reddit r/WallStreetBets posts are
filtered by stock mention, turned into a 7-element feature vector per post
(post_score, total_comms_num, len_of_post, positive/neutral/negative
sentiment, bias) and labelled with the next-day price move from Yahoo Finance.
"""
import os
import re

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
POSTS_CSV = os.path.join(DATA_DIR, "reddit_wsb.csv")

# Stocks studied in the report (Table 1) and the patterns used to detect them.
STOCK_PATTERNS = {
    "GME": r"\bgme\b|gamestop",
    "AMC": r"\bamc\b",
    "TSLA": r"\btsla\b|tesla",
    "FB": r"\bfb\b|facebook",
}
FEATURES = [
    "post_score", "total_comms_num", "len_of_post",
    "positive_sentiment", "neutral_sentiment", "negative_sentiment", "bias",
]

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"[@u/]*u/\w+|@\w+")
_SPECIAL = re.compile(r"[^a-z0-9\s]")
_SINGLE = re.compile(r"\b\w\b")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Preprocessing from Section 3: lowercase; drop handles, URLs, special
    characters and single characters; collapse repeated spaces."""
    text = _URL.sub(" ", str(text).lower())
    text = _HANDLE.sub(" ", text)
    text = _SPECIAL.sub(" ", text)
    text = _SINGLE.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


_analyzer = SentimentIntensityAnalyzer()


def sentiment(text: str) -> tuple:
    s = _analyzer.polarity_scores(text)
    return s["pos"], s["neu"], s["neg"]


def load_posts(path: str = POSTS_CSV) -> pd.DataFrame:
    ensure_data(path)
    cache = os.path.join(DATA_DIR, "posts_processed.pkl")   # VADER on 53k posts takes ~1 min
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return pd.read_pickle(cache)
    df = _load_posts_uncached(path)
    df.to_pickle(cache)
    return df


def ensure_data(path: str = POSTS_CSV) -> str:
    """Download the Kaggle posts CSV into data/ if it is not there yet (public dataset, no login)."""
    if not os.path.exists(path):
        import glob
        import shutil

        import kagglehub
        os.makedirs(os.path.dirname(path), exist_ok=True)
        src = glob.glob(os.path.join(kagglehub.dataset_download("gpreda/reddit-wallstreetsbets-posts"), "*.csv"))[0]
        shutil.copy(src, path)
    return path


def load_stock_posts(ticker: str, path: str = POSTS_CSV) -> pd.DataFrame:
    """Same result as posts_for_stock(load_posts(), ticker) but runs sentiment only on
    that stock's posts, so it takes seconds instead of ~1 minute (used by the quick demo)."""
    return _load_posts_uncached(ensure_data(path), ticker)


def _load_posts_uncached(path: str, ticker: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "timestamp"]).copy()   # NaN titles/timestamps dropped
    df["body"] = df["body"].fillna("")                     # body is empty for link posts
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["text"] = (df["title"] + " " + df["body"]).map(clean_text)
    df = df[df["text"].str.len() > 0]
    if ticker is not None:
        df = df[df["text"].str.contains(STOCK_PATTERNS[ticker], regex=True)].copy()
    pos_neu_neg = df["text"].map(sentiment)
    df["positive_sentiment"] = [s[0] for s in pos_neu_neg]
    df["neutral_sentiment"] = [s[1] for s in pos_neu_neg]
    df["negative_sentiment"] = [s[2] for s in pos_neu_neg]
    df["post_score"] = df["score"]
    df["total_comms_num"] = df["comms_num"]
    df["len_of_post"] = df["text"].str.split().str.len()
    df["bias"] = 1.0
    return df.sort_values("timestamp").reset_index(drop=True)


def posts_for_stock(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    mask = df["text"].str.contains(STOCK_PATTERNS[ticker], regex=True)
    return df[mask].copy()


def load_prices(ticker: str, start="2020-09-01", end="2021-09-15") -> pd.DataFrame:
    """Daily OHLC from Yahoo Finance (yfinance), cached to data/."""
    cache = os.path.join(DATA_DIR, f"prices_{ticker}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf
    yahoo = {"FB": "META"}.get(ticker, ticker)   # Facebook is listed as META on Yahoo now
    px = yf.download(yahoo, start=start, end=end, progress=False, auto_adjust=False)
    px.columns = [c[0] if isinstance(c, tuple) else c for c in px.columns]
    px = px[["Open", "High", "Low", "Close", "Volume"]].dropna()
    px.to_csv(cache)
    return px


def add_labels(posts: pd.DataFrame, prices: pd.DataFrame, mode: str = "paper") -> pd.DataFrame:
    """Attach the entry day k, next-open return and trading-day keys.

    mode="paper":    k = first trading day on/after the post's date; label
                     compares open(k+1) with open(k)  (P_k < P_{k+1}, as in the report).
    mode="tradable": k = first trading-day open that is *after* the post
                     (13:30 UTC cut-off), so the label is the move a trader could
                     actually capture after reading the post (no look-ahead).
    Returns rows with columns day, ret (open(k+1)/open(k) - 1), up (0/1).
    """
    opens = prices["Open"]
    days = opens.index.values
    stamps = posts["timestamp"]
    keyed = stamps.dt.normalize()
    if mode == "tradable":
        # a post made at/after 13:30 UTC can only be traded at the next open
        after_open = stamps >= keyed + pd.Timedelta(hours=13, minutes=30)
        keyed = keyed + pd.to_timedelta(after_open.astype(int), unit="D")
    idx = np.searchsorted(days, keyed.values.astype("datetime64[ns]"), side="left")
    ok = idx < len(days) - 1
    out = posts.loc[ok].copy()
    idx = idx[ok]
    out["day"] = days[idx]
    out["ret"] = opens.values[idx + 1] / opens.values[idx] - 1.0
    out["up"] = (out["ret"] > 0).astype(int)
    return out


def chrono_split(df: pd.DataFrame, fracs=(0.8, 0.1, 0.1)):
    """80/10/10 train/val/test split, in time order (no shuffling) so the test
    set is strictly later than training - posts on the same day share a label,
    so a random split would leak."""
    df = df.sort_values("timestamp")
    n = len(df)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def log_scale(X: pd.DataFrame) -> np.ndarray:
    """Heavy-tailed count features are log1p-transformed before standardising."""
    X = X[FEATURES].astype(float).copy()
    for c in COUNTS:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.values


COUNTS = ["post_score", "total_comms_num", "len_of_post"]
RANK_WINDOW = 200


def relative_features(posts: pd.DataFrame, window: int = RANK_WINDOW) -> pd.DataFrame:
    """Drift-robust version of the count features (used by the shrinkage ensemble).

    Raw engagement drifts with the size of the subreddit (GME's mean log post_score
    goes from 3.2 in the squeeze-month training split to 5.8 in the test split), so
    each count is replaced by its percentile rank among the stock's last `window`
    posts (this one included). Only earlier posts are used, so there is no look-ahead."""
    out = posts.sort_values("timestamp").copy()
    for c in COUNTS:
        out[c] = out[c].astype(float).clip(lower=0).rolling(window, min_periods=1).rank(pct=True)
    return out


def relative_rank(history, value, window: int = RANK_WINDOW) -> float:
    """Percentile rank of a new post's count among the last window-1 posts in `history`,
    computed exactly as relative_features does (used by the demo to score one post)."""
    s = pd.Series(np.append(np.asarray(history, float)[-(window - 1):], max(float(value), 0.0)))
    return float(s.rank(pct=True).iloc[-1])


def feature_matrix(X: pd.DataFrame) -> np.ndarray:
    """The 7 features as they are (relative features are already on a 0-1 scale)."""
    return X[FEATURES].astype(float).values
