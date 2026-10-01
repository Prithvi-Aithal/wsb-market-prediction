"""Does a *random* train/test split inflate accuracy?

Posts about the same stock on the same day share one label, so shuffling posts
puts near-duplicates of every test example into the training set. This script
repeats the report's direction task with (a) a shuffled 80/10/10 split and
(b) our chronological split, using the report's model configuration.

    python src/leakage_check.py      ->  results/leakage_check.csv
"""
import os

import pandas as pd
from joblib import Parallel, delayed

import wsb_pipeline as wp
from models import NewtonLogisticRegression, fit_neural_net
from run_experiments import BATCH, RES, TICKERS, make_xy


def run(ticker, split, seed, posts, prices):
    lab = wp.add_labels(posts, prices, "paper")
    if split == "shuffled":
        lab = lab.sample(frac=1.0, random_state=seed)
    n = len(lab); a, b = int(.8 * n), int(.9 * n)
    tr, va, te = lab.iloc[:a], lab.iloc[a:b], lab.iloc[b:]
    if split == "chronological":
        tr, va, te = wp.chrono_split(lab)
    Xtr, sc = make_xy(tr); Xva, _ = make_xy(va, sc); Xte, _ = make_xy(te, sc)
    ytr, yva, yte = tr["up"].values, va["up"].values, te["up"].values
    lr = NewtonLogisticRegression().fit(Xtr, ytr)
    net, _ = fit_neural_net(Xtr, ytr, Xva, yva, batch_size=BATCH.get(ticker, 16), seed=seed)
    return {"stock": ticker, "split": split, "seed": seed,
            "lr_acc": float((lr.predict(Xte) == yte).mean()),
            "nn_acc": float((net.predict(Xte) == yte).mean()),
            "majority_acc": float((yte == int(ytr.mean() >= .5)).mean())}


if __name__ == "__main__":
    posts = wp.load_posts()
    sp = {t: wp.posts_for_stock(posts, t) for t in TICKERS}
    px = {t: wp.load_prices(t) for t in TICKERS}
    jobs = [(t, s, seed) for t in TICKERS for s in ("shuffled", "chronological") for seed in range(3)]
    rows = Parallel(n_jobs=-1)(delayed(run)(t, s, seed, sp[t], px[t]) for t, s, seed in jobs)
    df = pd.DataFrame(rows).groupby(["stock", "split"]).mean(numeric_only=True).drop(columns="seed").round(3)
    df.to_csv(os.path.join(RES, "leakage_check.csv"))
    print(df)
