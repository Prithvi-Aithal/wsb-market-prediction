"""Live demo: score a WallStreetBets post with the trained models.

    python src/demo.py --stock GME --title "GME to the moon, diamond hands" --score 1200 --comments 340
    python src/demo.py --stock TSLA --task volatility --title "Tesla puts printing"
    python src/demo.py --replay GME        # replay the daily strategy on the held-out test period

Run `python src/run_experiments.py` first so that models/ exists.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

import wsb_pipeline as wp


def load(stock, task):
    mode = "tradable" if task == "direction" else "paper"
    return joblib.load(os.path.join(wp.ROOT, "models", f"{stock}_{mode}_{task}.joblib"))


def featurize(title, body, score, comments):
    text = wp.clean_text(title + " " + body)
    pos, neu, neg = wp.sentiment(text)
    row = pd.DataFrame([{
        "post_score": score, "total_comms_num": comments, "len_of_post": len(text.split()),
        "positive_sentiment": pos, "neutral_sentiment": neu, "negative_sentiment": neg, "bias": 1.0}])
    return row, (pos, neu, neg)


def predict(args):
    m = load(args.stock, args.task)
    row, (pos, neu, neg) = featurize(args.title, args.body, args.score, args.comments)
    x = m["scaler"].transform(wp.log_scale(row))
    p_nn = float(m["nn"].predict_proba(x)[0, 1])
    p_lr = float(1 / (1 + np.exp(-x @ m["lr_coef"]))[0])
    what = "price RISES at next open" if args.task == "direction" else "next-day move is LARGE (volatile)"
    print(f"Sentiment  pos={pos:.2f} neu={neu:.2f} neg={neg:.2f}")
    print(f"P({what}):  neural net={p_nn:.2f}   logistic regression={p_lr:.2f}")
    if args.task == "direction":
        print("Signal:", "BUY / long" if p_nn >= .5 else "stay flat")
    else:
        print("Signal:", "VOLATILE - size down" if p_nn >= .5 else "calm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="GME", choices=list(wp.STOCK_PATTERNS))
    ap.add_argument("--task", default="direction", choices=["direction", "volatility"])
    ap.add_argument("--title", default="GME to the moon, diamond hands")
    ap.add_argument("--body", default="")
    ap.add_argument("--score", type=int, default=100)
    ap.add_argument("--comments", type=int, default=20)
    ap.add_argument("--replay", metavar="STOCK", choices=list(wp.STOCK_PATTERNS))
    a = ap.parse_args()
    if a.replay:
        bt = pd.read_csv(os.path.join(wp.ROOT, "results", "backtest.csv"))
        print(bt[bt.stock == a.replay].T.to_string(header=False))
    else:
        predict(a)
