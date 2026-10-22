"""Live demo: score a WallStreetBets post with the trained models.

    python src/demo.py --stock GME --title "GME to the moon, diamond hands" --score 1200 --comments 340
    python src/demo.py --stock TSLA --task volatility --title "Tesla puts printing"
    python src/demo.py --replay GME        # replay the daily strategy on the held-out test period
    python src/demo.py --quick GME         # run the real pipeline live on one stock (a few minutes for GME)

Scoring a post and --replay need the saved models/results (already in the repo, or created by
`python src/run_experiments.py`). --quick needs only the dataset (downloaded automatically).
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


def ensemble_input(m, row):
    """Scaled ensemble features for one post: counts become percentile ranks against the
    stock's latest posts (wp.relative_rank), as in training."""
    rel = row.copy()
    for j, c in enumerate(wp.COUNTS):
        rel[c] = wp.relative_rank(m["ens_history"][:, j], rel[c].iloc[0])
    return m["ens_scaler"].transform(wp.feature_matrix(rel))


def predict(args):
    m = load(args.stock, args.task)
    row, (pos, neu, neg) = featurize(args.title, args.body, args.score, args.comments)
    x = m["scaler"].transform(wp.log_scale(row))
    p_nn = float(m["nn"].predict_proba(x)[0, 1])
    p_lr = float(1 / (1 + np.exp(-x @ m["lr_coef"]))[0])
    ens = m["ens"]
    p_ens = float(ens.predict_proba(ensemble_input(m, row))[0])
    what = "price RISES at next open" if args.task == "direction" else "next-day move is LARGE"
    print(f"Sentiment  pos={pos:.2f} neu={neu:.2f} neg={neg:.2f}")
    print(f"P({what}):  neural net={p_nn:.2f}   logistic regression={p_lr:.2f}   "
          f"shrinkage ensemble={p_ens:.2f} (lam={ens.lam}, prior={ens.prior_:.2f})")
    if args.task == "direction":
        print("Signal (ensemble):", "BUY / long" if p_ens >= .5 else "stay flat")
    else:
        print("Signal (ensemble):", "VOLATILE - size down" if p_ens >= .5 else "calm")


def quick(stock, use_nn=True):
    """Live end-to-end run for one stock: raw posts -> features -> labels -> chronological
    split -> Newton logistic regression (+ neural net) + shrinkage ensemble -> test accuracy vs baselines."""
    import time

    from sklearn.metrics import f1_score

    from models import ENS_LAMS, NewtonLogisticRegression, fit_neural_net
    from run_experiments import BATCH, fit_ensemble, make_xy

    t0 = time.time()
    step = lambda msg: print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)
    step(f"loading + cleaning posts, VADER sentiment for {stock} posts ...")
    posts = wp.load_stock_posts(stock)
    step(f"{len(posts):,} posts mention {stock}; features = {wp.FEATURES}")
    prices = wp.load_prices(stock)
    step(f"prices: {len(prices)} trading days ({prices.index[0].date()} to {prices.index[-1].date()})")
    for task, mode in (("direction", "tradable"), ("volatility", "paper")):
        lab = wp.add_labels(posts, prices, mode)
        tr, va, te = wp.chrono_split(lab)
        if task == "volatility":
            thr = tr["ret"].abs().median()
            for d in (tr, va, te):
                d["y"] = (d["ret"].abs() > thr).astype(int)
        else:
            for d in (tr, va, te):
                d["y"] = d["up"]
        print(f"\n=== {task} task ({mode} labels): train {len(tr)} / val {len(va)} / test {len(te)} "
              f"(test = {te.timestamp.min().date()} to {te.timestamp.max().date()}, strictly after training)")
        Xtr, sc = make_xy(tr); Xva, _ = make_xy(va, sc); Xte, _ = make_xy(te, sc)
        ytr, yva, yte = tr["y"].values, va["y"].values, te["y"].values
        maj = int(ytr.mean() >= .5)
        print(f"baselines on test: random guess 0.500 | majority class ({maj}) {(yte == maj).mean():.3f}")
        lr = NewtonLogisticRegression().fit(Xtr, ytr)
        print("Newton logistic regression, ||delta w|| per iteration:",
              " ".join(f"{d:.1e}" for d in lr.deltas_))
        print("  weights", {f: round(float(c), 2) for f, c in zip(wp.FEATURES, lr.coef_)})
        print(f"  test accuracy {(lr.predict(Xte) == yte).mean():.3f}  F1 {f1_score(yte, lr.predict(Xte), zero_division=0):.3f}")
        if use_nn:
            step("training 3x256 ReLU network (Adam, 30 epochs, best epoch on validation) ...")
            net, hist = fit_neural_net(Xtr, ytr, Xva, yva, batch_size=BATCH.get(stock, 16), seed=0)
            p = net.predict(Xte)
            print(f"  neural net test accuracy {(p == yte).mean():.3f}  F1 {f1_score(yte, p, zero_division=0):.3f} "
                  f"(best validation epoch {int(np.argmax(hist)) + 1}, val acc {max(hist):.3f})")
        step("tuning shrinkage ensemble (LR + random forest + gradient boosting + naive Bayes; "
             "random search + lam by forward-chaining CV on train+val) ...")
        ens, _, tuned, p_ens = fit_ensemble(posts, tr, va, te)
        p = (p_ens >= .5).astype(int)
        print(f"  {tuned['n_folds']}-fold forward-chaining CV log-loss of the best config for each lam:")
        print("   ", "  ".join(f"lam={l}: {v:.3f}" for l, v in zip(ENS_LAMS, tuned["curve"])))
        print(f"  chosen lam={ens.lam} (0 = trust the prior, 1 = trust the features), prior={ens.prior_:.2f}, cfg {tuned['cfg']}")
        print(f"  shrinkage ensemble test accuracy {(p == yte).mean():.3f}  F1 {f1_score(yte, p, zero_division=0):.3f}")
    step("done. Compare each model with the majority-class baseline, not with 0.5.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="GME", choices=list(wp.STOCK_PATTERNS))
    ap.add_argument("--task", default="direction", choices=["direction", "volatility"])
    ap.add_argument("--title", default="GME to the moon, diamond hands")
    ap.add_argument("--body", default="")
    ap.add_argument("--score", type=int, default=100)
    ap.add_argument("--comments", type=int, default=20)
    ap.add_argument("--replay", metavar="STOCK", choices=list(wp.STOCK_PATTERNS))
    ap.add_argument("--quick", metavar="STOCK", choices=list(wp.STOCK_PATTERNS),
                    help="run the full pipeline live for one stock")
    ap.add_argument("--no-nn", action="store_true", help="with --quick: skip the neural net (faster)")
    a = ap.parse_args()
    if a.quick:
        quick(a.quick, use_nn=not a.no_nn)
    elif a.replay:
        bt = pd.read_csv(os.path.join(wp.ROOT, "results", "backtest.csv"))
        print(bt[bt.stock == a.replay].fillna("n/a (no trades)").T.to_string(header=False))
    else:
        predict(a)
