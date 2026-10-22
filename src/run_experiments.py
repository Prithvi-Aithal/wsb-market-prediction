"""Run every experiment in the report and write tables/figures to results/.

    python src/run_experiments.py

Label modes (see wsb_pipeline.add_labels):  paper | tradable
Tasks:  direction  - does the price rise from open(k) to open(k+1)?
        volatility - is |open(k+1)/open(k) - 1| larger than the training-set
                     median move for that stock (a "significant shift")?
Models: Newton logistic regression and the 3x256 network from the report, plus our
        shrinkage ensemble (models.ShrinkageEnsemble) on drift-robust relative features.
"""
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wsb_pipeline as wp
from models import NewtonLogisticRegression, ShrinkageEnsemble, fit_neural_net, tune_ensemble

ROOT = wp.ROOT
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

TICKERS = list(wp.STOCK_PATTERNS)
SEEDS = range(5)
BATCH = {"TSLA": 64, "GME": 64}      # report: batch 64 for TSLA/GME, 16 elsewhere
CYAN = "#00bfbf"
plt.style.use("seaborn-v0_8-whitegrid")


L2_GRID = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]


def random_configs(rng, n=6):
    """A report-like configuration first (3x256, lr 1e-3, weight decay 1e-4, batch 64; searched for
    15 epochs), then random draws from the search space."""
    yield {"hidden": (256, 256, 256), "lr": 1e-3, "alpha": 1e-4, "batch_size": 64}
    for _ in range(n - 1):
        yield {"hidden": [(32,), (64, 64), (128, 128), (256, 256), (256, 256, 256)][rng.integers(5)],
               "lr": float(rng.choice([1e-4, 3e-4, 1e-3, 3e-3])),
               "alpha": float(rng.choice([1e-4, 1e-2, 1e-1, 1.0])),
               "batch_size": int(rng.choice([32, 64, 128]))}


def make_xy(split, scaler=None, transform=wp.log_scale):
    X = transform(split)
    if scaler is None:
        scaler = StandardScaler().fit(X)
        scaler.mean_[-1], scaler.scale_[-1] = 0.0, 1.0   # keep bias column == 1
    return scaler.transform(X), scaler


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None   # undefined with one class


def fit_ensemble(posts, tr, va, te):
    """New model: shrinkage ensemble on relative (percentile-rank) features. Hyper-parameters
    and lam come from forward-chaining CV on train+val; the chosen model is then refit on
    train+val (the most recent labelled data) and applied to the test split."""
    rel = wp.relative_features(posts)
    Rtr, sc = make_xy(rel.loc[tr.index], transform=wp.feature_matrix)
    Rva, _ = make_xy(rel.loc[va.index], sc, transform=wp.feature_matrix)
    Rte, _ = make_xy(rel.loc[te.index], sc, transform=wp.feature_matrix)
    X, y = np.vstack([Rtr, Rva]), np.concatenate([tr["y"].values, va["y"].values])
    cfg, lam, cv_ll = tune_ensemble(X, y, len(tr))
    ens = ShrinkageEnsemble(lam=lam, **cfg).fit(X, y)
    return ens, sc, cfg, cv_ll, ens.predict_proba(Rte)


def evaluate(ticker, mode, task, posts, prices, save=False):
    lab = wp.add_labels(posts, prices, mode)
    tr, va, te = wp.chrono_split(lab)
    if task == "volatility":
        thr = tr["ret"].abs().median()
        for d in (tr, va, te):
            d["y"] = (d["ret"].abs() > thr).astype(int)
    else:
        for d in (tr, va, te):
            d["y"] = d["up"]
    Xtr, sc = make_xy(tr)
    Xva, _ = make_xy(va, sc)
    Xte, _ = make_xy(te, sc)
    ytr, yva, yte = tr["y"].values, va["y"].values, te["y"].values

    out = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "test_pos_rate": float(yte.mean())}
    # baselines: uniform random guess (expected accuracy, 500 draws) and majority class
    rng = np.random.default_rng(0)
    out["random_acc"] = float(np.mean([(rng.integers(0, 2, len(yte)) == yte).mean() for _ in range(500)]))
    maj = int(ytr.mean() >= 0.5)
    out["majority_acc"] = float((yte == maj).mean())

    # logistic regression (Newton's method) - report setting (l2=1e-4) ...
    lr = NewtonLogisticRegression().fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)
    # ... and tuned: L2 strength chosen by validation log-loss
    best_l2, best_ll = None, np.inf
    for l2 in L2_GRID:
        m = NewtonLogisticRegression(l2=l2).fit(Xtr, ytr)
        pv = np.clip(m.predict_proba(Xva), 1e-9, 1 - 1e-9)
        ll = -np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv))
        if ll < best_ll:
            best_ll, best_l2 = ll, l2
    lr_t = NewtonLogisticRegression(l2=best_l2).fit(Xtr, ytr)
    out["lr_tuned_l2"] = best_l2
    out["lr_tuned_acc"] = float(((lr_t.predict_proba(Xte) >= .5) == yte).mean())
    out["lr_acc"] = float(((p_lr >= .5) == yte).mean())
    out["lr_f1"] = float(f1_score(yte, p_lr >= .5, zero_division=0))
    out["lr_converged"] = bool(lr.converged_)
    out["lr_iters"] = len(lr.deltas_)

    # neural network, averaged over seeds
    accs, f1s, probs, nets = [], [], [], []
    for s in SEEDS:
        net, _ = fit_neural_net(Xtr, ytr, Xva, yva, batch_size=BATCH.get(ticker, 16), seed=s)
        p = net.predict_proba(Xte)[:, 1]
        probs.append(p); nets.append(net)
        accs.append(float(((p >= .5) == yte).mean()))
        f1s.append(float(f1_score(yte, p >= .5, zero_division=0)))
    out["nn_acc"], out["nn_acc_std"] = float(np.mean(accs)), float(np.std(accs))
    out["nn_f1"] = float(np.mean(f1s))
    p_nn = np.mean(probs, axis=0)
    out["nn_ens_acc"] = float(((p_nn >= .5) == yte).mean())
    # hyper-parameter search for the net (random search, selected on validation
    # accuracy with 1 seed / 15 epochs; the winner is re-trained with 5 seeds)
    search_rng = np.random.default_rng(42)
    best_cfg, best_val = None, -1.0
    for cfg in random_configs(search_rng):
        net, hist = fit_neural_net(Xtr, ytr, Xva, yva, epochs=15, seed=0, **cfg)
        if max(hist) > best_val:
            best_val, best_cfg = max(hist), cfg
    out["nn_tuned_cfg"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items()}
    out["nn_tuned_val_acc"] = float(best_val)
    taccs, tprobs, tnets = [], [], []
    for s in SEEDS:
        net, _ = fit_neural_net(Xtr, ytr, Xva, yva, seed=s, **best_cfg)
        p = net.predict_proba(Xte)[:, 1]
        tprobs.append(p); tnets.append(net)
        taccs.append(float(((p >= .5) == yte).mean()))
    out["nn_tuned_acc"], out["nn_tuned_acc_std"] = float(np.mean(taccs)), float(np.std(taccs))
    p_tuned = np.mean(tprobs, axis=0)
    out["nn_tuned_ens_acc"] = float(((p_tuned >= .5) == yte).mean())
    out["se"] = float(np.sqrt(0.25 / max(len(yte), 1)))   # binomial std. error at p=0.5

    # new model: shrinkage ensemble (tuned by forward-chaining CV, never on test)
    ens, ens_sc, ens_cfg, cv_ll, p_ens = fit_ensemble(posts, tr, va, te)
    out["ens_acc"] = float(((p_ens >= .5) == yte).mean())
    out["ens_f1"] = float(f1_score(yte, p_ens >= .5, zero_division=0))
    out["ens_lam"], out["ens_cfg"], out["ens_cv_logloss"] = ens.lam, ens_cfg, cv_ll
    # probabilistic metrics: log-loss (lower is better) and ROC AUC (0.5 = no ranking signal)
    p_lr_t = lr_t.predict_proba(Xte)
    for k, p in (("majority", np.full(len(yte), ytr.mean())), ("lr_tuned", p_lr_t),
                 ("nn_tuned", p_tuned), ("ens", p_ens)):
        out[f"{k}_logloss"] = float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        if k != "majority":
            out[f"{k}_auc"] = auc(yte, p)

    if save:
        joblib.dump({"scaler": sc, "lr_coef": lr_t.coef_, "nn": tnets[0], "nn_cfg": best_cfg,
                     "ens": ens, "ens_scaler": ens_sc,
                     # last posts' raw counts, to rank a new post against (wp.relative_rank)
                     "ens_history": posts.sort_values("timestamp")[wp.COUNTS].tail(wp.RANK_WINDOW - 1).values},
                    os.path.join(MODELS, f"{ticker}_{mode}_{task}.joblib"))
    te = te.assign(p_nn=p_tuned, p_lr=p_lr, p_ens=p_ens)
    return out, lr, te


def backtest(te, col="p_nn"):
    """Long when the majority of that day's posts are predicted 'up', else flat.
    Benchmark: long on every day that had posts (same days, no model)."""
    g = te.groupby("day").agg(ret=("ret", "first"), frac_up=(col, lambda p: (p >= .5).mean()),
                              n=("ret", "size"))
    g["strategy"] = np.where(g["frac_up"] > 0.5, g["ret"], 0.0)
    return g


def main():
    print("Loading and preprocessing posts ...")
    posts = wp.load_posts()
    prices = {t: wp.load_prices(t) for t in TICKERS}
    stock_posts = {t: wp.posts_for_stock(posts, t) for t in TICKERS}
    mentions = {t: len(v) for t, v in stock_posts.items()}
    print("mentions:", mentions)

    results, curves, bt_rows, equity = {}, {}, [], {}
    jobs = [(m, k, t) for m in ("paper", "tradable") for k in ("direction", "volatility") for t in TICKERS]

    def run(mode, task, t):
        save = (mode == "tradable" and task == "direction") or (mode == "paper" and task == "volatility")
        r, lr, te = evaluate(t, mode, task, stock_posts[t], prices[t], save=save)
        return r, lr.deltas_, te

    outs = Parallel(n_jobs=-1, verbose=5)(delayed(run)(*j) for j in jobs)   # one process per setting
    for (mode, task, t), (r, deltas, te) in zip(jobs, outs):
        results[f"{mode}/{task}/{t}"] = r
        if mode == "paper" and task == "volatility":
            curves[t] = deltas
        if mode == "tradable" and task == "direction":
            g, ge = backtest(te), backtest(te, "p_ens")
            equity[t] = g.assign(ensemble=ge["strategy"])
            bt_rows.append({
                "stock": t, "days_traded": int((g["frac_up"] > .5).sum()), "days_with_posts": len(g),
                "strategy_return_%": 100 * float((1 + g["strategy"]).prod() - 1),
                "always_long_return_%": 100 * float((1 + g["ret"]).prod() - 1),
                "strategy_hit_rate": float((g.loc[g["frac_up"] > .5, "ret"] > 0).mean()) if (g["frac_up"] > .5).any() else None,   # None = no trades
                "ensemble_days_traded": int((ge["frac_up"] > .5).sum()),
                "ensemble_return_%": 100 * float((1 + ge["strategy"]).prod() - 1),
            })
        print(f"{mode:8s} {task:10s} {t:5s}", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump({"mentions": mentions, "results": results, "backtest": bt_rows}, f, indent=2)
    rows = [{"setting": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(RES, "results.csv"), index=False)
    pd.DataFrame(bt_rows).to_csv(os.path.join(RES, "backtest.csv"), index=False)

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(mentions.keys(), mentions.values(), color=CYAN)
    ax.set_ylabel("posts mentioning stock"); ax.set_title("Mentions of each stock in r/WallStreetBets")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_mentions.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for t, d in curves.items():
        ax.semilogy(range(1, len(d) + 1), np.maximum(d, 1e-12), marker="o", label=t)
    ax.set_xlabel("Newton iteration"); ax.set_ylabel("||delta w||")
    ax.set_title("Convergence of the weight vector (logistic regression)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_convergence.png"), dpi=200); plt.close(fig)

    for task in ("direction", "volatility"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, mode in zip(axes, ("paper", "tradable")):
            x = np.arange(len(TICKERS)); w = 0.115
            for i, (key, lab) in enumerate([("nn_acc", "NN (report cfg)"), ("nn_tuned_acc", "NN (tuned)"),
                                            ("lr_acc", "LogReg (report)"), ("lr_tuned_acc", "LogReg (tuned)"),
                                            ("ens_acc", "Shrinkage ensemble (new)"),
                                            ("majority_acc", "Majority"), ("random_acc", "Random")]):
                vals = [results[f"{mode}/{task}/{t}"][key] for t in TICKERS]
                err = [results[f"{mode}/{task}/{t}"].get(key + "_std", 0) if key.startswith("nn") else 0 for t in TICKERS]
                ax.bar(x + (i - 3) * w, vals, w, yerr=err, label=lab)
            ax.axhline(0.5, color="k", lw=.6, ls=":")
            ax.set_xticks(x); ax.set_xticklabels(TICKERS); ax.set_title(f"{task} - '{mode}' labels")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("test accuracy")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7.5, frameon=False)   # below the bars, never covering them
        fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(RES, f"fig_accuracy_{task}.png"), dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, len(TICKERS), figsize=(12, 2.8))
    for ax, t in zip(axes, TICKERS):
        g = equity[t]
        ax.plot(g.index, (1 + g["strategy"]).cumprod(), label="WSB-NN strategy", color=CYAN)
        ax.plot(g.index, (1 + g["ensemble"]).cumprod(), label="ensemble strategy", color="#d1495b", ls="--")
        ax.plot(g.index, (1 + g["ret"]).cumprod(), label="always long", color="gray")
        ax.set_title(t); ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].legend(fontsize=7); axes[0].set_ylabel("growth of $1")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_backtest.png"), dpi=200); plt.close(fig)
    print("\nBacktest:\n", pd.DataFrame(bt_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
