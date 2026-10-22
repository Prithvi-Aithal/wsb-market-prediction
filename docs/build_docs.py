"""Build the 2-page write-up (PDF) and the presentation (PPTX) from results/.

    python docs/build_docs.py

The explanatory figures for the new model (docs/fig_*.png) are drawn here from
results/results.json; the worked example scores the demo post with the saved model.
"""
import json
import math
import os
import sys

import joblib
import matplotlib
import pandas as pd
from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
import demo                                   # noqa: E402  (featurize / ensemble_input for the worked example)
from models import ENS_LAMS                   # noqa: E402

R = json.load(open(os.path.join(RES, "results.json")))
res, mentions, bt = R["results"], R["mentions"], R["backtest"]
leak = pd.read_csv(os.path.join(RES, "leakage_check.csv"))
T = ["GME", "AMC", "TSLA", "FB"]
CYAN, RED, GREY = "#00bfbf", "#d1495b", "#8c8c8c"
plt.style.use("seaborn-v0_8-whitegrid")


def row(task, mode, t, keys):
    r = res[f"{mode}/{task}/{t}"]
    return [t, str(r["n_test"])] + [f"{r[k]:.3f}" for k in keys]


KEYS = ["nn_acc", "nn_tuned_acc", "lr_acc", "lr_tuned_acc", "ens_acc", "majority_acc", "random_acc"]
HEAD = ["Stock", "n test", "NN\n(report)", "NN\n(tuned)", "LogReg\n(report)", "LogReg\n(tuned)", "Ensemble\n(new)",
        "Majority", "Random"]


def table_data(task, mode):
    return [HEAD] + [row(task, mode, t, KEYS) for t in T]


# summary of the new model over all 16 settings (2 label modes x 2 tasks x 4 stocks)
SETTINGS = [f"{m}/{k}/{t}" for m in ("paper", "tradable") for k in ("direction", "volatility") for t in T]
OLD = ["nn_tuned_acc", "lr_tuned_acc"]
E = {
    "n": len(SETTINGS),
    "match_maj": sum(res[k]["ens_acc"] >= res[k]["majority_acc"] - 1e-9 for k in SETTINGS),
    "old_match_maj": sum(max(res[k][o] for o in OLD) >= res[k]["majority_acc"] - 1e-9 for k in SETTINGS),
    "beat_old": sum(res[k]["ens_acc"] >= max(res[k][o] for o in OLD) - 1e-9 for k in SETTINGS),
    "acc": {k: sum(res[s][k] for s in SETTINGS) / len(SETTINGS) for k in OLD + ["ens_acc", "majority_acc"]},
    "ll": {k: sum(res[s][f"{k}_logloss"] for s in SETTINGS) / len(SETTINGS) for k in ("nn_tuned", "lr_tuned", "ens", "majority")},
    "lam0": sum(res[s]["ens_lam"] == 0 for s in SETTINGS),
    "lams": {lam: sum(res[s]["ens_lam"] == lam for s in SETTINGS) for lam in ENS_LAMS},
    "below": [s for s in SETTINGS if res[s]["ens_acc"] < res[s]["majority_acc"] - 1e-9],
    "short_folds": sorted({s.split("/")[1] + " " + s.split("/")[2] for s in SETTINGS if res[s]["ens_cv_folds"] < 4}),
}
VOL = {t: res[f"paper/volatility/{t}"] for t in T}       # drift numbers (same posts for every setting of a stock)


def gain(key):
    r = res[key]
    return f"{r['ens_acc']:.2f} vs {max(r[o] for o in OLD):.2f}"


def nice(setting):
    mode, task, t = setting.split("/")
    return f"{t} {task} ({mode})"


# ------------------------------------------------------------------ worked example (the demo post)
EX_POST = dict(title="GME to the moon, diamond hands", body="", score=1200, comments=340)
EX_MODEL = joblib.load(os.path.join(ROOT, "models", "GME_tradable_direction.joblib"))


def worked_example():
    ens = EX_MODEL["ens"]
    row_, _ = demo.featurize(EX_POST["title"], EX_POST["body"], EX_POST["score"], EX_POST["comments"])
    x = demo.ensemble_input(EX_MODEL, row_)
    z = x[:, :-1]
    parts = [("Newton logistic regression", float(ens.lr_.predict_proba(x)[0])),
             ("Random forest", float(ens.rf_.predict_proba(z)[0, 1])),
             ("Gradient boosting", float(ens.gb_.predict_proba(z)[0, 1])),
             ("Gaussian naive Bayes", float(ens.nb_.predict_proba(z)[0, 1]))]
    return parts, float(ens.vote_proba(x)[0]), ens.prior_, ens.lam


EX_PARTS, EX_VOTE, EX_PRIOR, EX_LAM = worked_example()


# ------------------------------------------------------------------ figures for the new model
def fig_path(name):
    return os.path.join(HERE, name)


def draw_drift(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = range(len(T))
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], [100 * VOL[t]["train_pos_rate"] for t in T], w, label="training", color=GREY)
    axes[0].bar([i + w / 2 for i in x], [100 * VOL[t]["test_pos_rate"] for t in T], w, label="test", color=RED)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(T); axes[0].set_ylabel("% of posts labelled 'volatile'")
    axes[0].set_title("Label shift: the market calmed down"); axes[0].legend(loc="upper center", bbox_to_anchor=(0.33, 1.0))
    w = 0.27
    for i, (k, c) in enumerate((("train", GREY), ("val", "#f0a202"), ("test", RED))):
        axes[1].bar([j + (i - 1) * w for j in x], [VOL[t]["mean_log_score"][k] for t in T], w, label=k, color=c)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(T); axes[1].set_ylabel("mean log(1 + post score)")
    axes[1].set_title("Feature drift: posts got far more upvotes"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#333333"))


def draw_pipeline(path):
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.2); ax.axis("off")
    _box(ax, 0.1, 1.55, 1.8, 1.1, "WSB post\ntext, score,\ncomments, length", "#eeeeee")
    _box(ax, 2.3, 1.35, 2.0, 1.5, "Features\nVADER sentiment\n+ score/comments/length\nas percentile rank\nvs last 200 posts", "#d9f2f2", 9)
    names = ["Newton logistic\nregression", "Random forest", "Gradient boosting", "Gaussian naive\nBayes"]
    ys = [3.15, 2.2, 1.25, 0.3]
    for n_, y in zip(names, ys):
        _box(ax, 4.8, y, 1.9, 0.75, n_, "#fde9c9", 9)
        _arrow(ax, (4.3, 2.1), (4.8, y + 0.37))
        _arrow(ax, (6.7, y + 0.37), (7.2, 2.1))
    _box(ax, 7.2, 1.6, 1.5, 1.0, "average\n= p_vote\n(soft vote)", "#fde9c9", 9)
    _box(ax, 9.1, 1.45, 1.9, 1.3, "shrink to prior\np = λ·p_vote\n+ (1−λ)·prior", "#f9d0d6", 10)
    _box(ax, 11.3, 1.6, 1.1, 1.0, "p ≥ 0.5\n→ predict 1", "#eeeeee", 9)
    _box(ax, 8.6, 0.05, 2.9, 0.9, "λ and model settings chosen by\nforward-chaining cross-validation", "#e4e4f7", 9)
    _arrow(ax, (1.9, 2.1), (2.3, 2.1)); _arrow(ax, (8.7, 2.1), (9.1, 2.1)); _arrow(ax, (11.0, 2.1), (11.3, 2.1))
    _arrow(ax, (10.05, 0.95), (10.05, 1.45))
    _box(ax, 9.1, 3.25, 1.9, 0.75, "prior = share of 1-labels\nin the training data", "#f9d0d6", 8.5)
    _arrow(ax, (10.05, 3.25), (10.05, 2.75))
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def draw_forward_cv(path):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    val_end = 1.125                                      # validation = 10 % of posts vs 80 % for training
    rows = [("Fold 1", 0.4, (0.4, 0.6)), ("Fold 2", 0.6, (0.6, 0.8)), ("Fold 3", 0.8, (0.8, 1.0)),
            ("Fold 4", 1.0, (1.0, val_end))]
    for i, (name, tr_end, (a, b)) in enumerate(rows):
        y = len(rows) - i
        ax.barh(y, tr_end, color=CYAN, height=0.6)
        ax.barh(y, b - a, left=a, color="#f0a202", height=0.6)
        ax.text(-0.02, y, name, ha="right", va="center", fontsize=10)
    ax.barh(0, val_end, color=CYAN, height=0.6, hatch="//", edgecolor="white")
    ax.barh(0, 0.125, left=val_end, color=RED, height=0.6)
    ax.text(-0.02, 0, "Final model", ha="right", va="center", fontsize=10, weight="bold")
    ax.axvline(1.0, color="k", lw=.8, ls=":"); ax.axvline(val_end, color="k", lw=.8, ls=":")
    ax.text(0.5, 4.75, "training split (oldest 80 %)", ha="center", fontsize=9)
    ax.text(1.0625, 4.75, "val", ha="center", fontsize=9)
    ax.text(1.1875, 4.75, "test", ha="center", fontsize=9)
    ax.set_xlim(-0.01, 1.26); ax.set_ylim(-0.6, 5.1); ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlabel("time  →")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CYAN, label="fit on (the past)"), Patch(color="#f0a202", label="score log-loss on (the future)"),
                       Patch(color=RED, label="test: used once, at the end")], loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


CURVES = ["paper/volatility/TSLA", "tradable/direction/TSLA", "paper/direction/GME", "tradable/volatility/AMC"]


def draw_lambda(path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, c in zip(CURVES, (CYAN, "#f0a202", RED, "#6a4c93")):
        curve, lam = res[k]["ens_cv_curve"], res[k]["ens_lam"]
        ax.plot(ENS_LAMS, curve, marker="o", color=c, label=f"{nice(k)}: chose λ = {lam}")
        ax.plot([lam], [curve[ENS_LAMS.index(lam)]], marker="*", ms=18, color=c, mec="k")
    ax.set_xlabel("λ   (0 = always the prior / majority,  1 = fully trust the 4 models)")
    ax.set_ylabel("mean CV log-loss (lower = better)")
    ax.set_title("Each setting picks the λ with the lowest forward-CV log-loss (big marker)")
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


FIGS = {"drift": fig_path("fig_drift.png"), "pipeline": fig_path("fig_ensemble_pipeline.png"),
        "cv": fig_path("fig_forward_cv.png"), "lambda": fig_path("fig_lambda_choice.png")}


def draw_figures():
    draw_drift(FIGS["drift"]); draw_pipeline(FIGS["pipeline"]); draw_forward_cv(FIGS["cv"]); draw_lambda(FIGS["lambda"])


# ------------------------------------------------------------------ PDF (2 pages)
def build_pdf(path):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=10.4, spaceAfter=2.2)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=10, spaceBefore=3, spaceAfter=1.5)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=14, spaceAfter=2)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.1 * cm,
                            title="Mini-project write-up: WSB market prediction",
                            author="Prithvi Aithal, Sagar A Walikar")
    s = []

    def tbl(data):
        t = Table(data, hAlign="CENTER", colWidths=[1.3 * cm, 1.3 * cm] + [1.85 * cm] * 7)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9f2f2")),
                               ("BACKGROUND", (6, 1), (6, -1), colors.HexColor("#fbe3e7")),
                               ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3)]))
        return t

    s.append(Paragraph("Predicting Market Volatility and Short-Term Trading Signals from Reddit's WallStreetBets", title))
    s.append(Paragraph("UE24CS352A Machine Learning - Mini-Project write-up &nbsp;|&nbsp; Team: Prithvi Aithal (PES1UG24CS909), Sagar A Walikar (PES1UG24CS396) "
                       "&nbsp;|&nbsp; Repository: github.com/Prithvi-Aithal/wsb-market-prediction (private)", ParagraphStyle("c", parent=body, alignment=1)))
    s.append(Paragraph("1. Problem statement", h))
    s.append(Paragraph(
        "Retail-investor communities such as r/WallStreetBets (WSB) moved prices during the January 2021 GameStop short squeeze. "
        "We ask whether the <b>posts themselves</b> contain a usable signal about (a) the <b>direction</b> of the next price move "
        "(will open(next day) exceed open(this day)?), (b) <b>volatility</b> (will the move be unusually large?), and (c) whether "
        "such predictions can drive a simple short-term trading strategy. We re-implement the CS229 report's approach, extend it "
        "with hyper-parameter tuning, stricter baselines, a leakage-free evaluation and a back-test, and add a <b>new model</b> "
        "(a drift-aware shrinkage ensemble, section 4) to improve on the report's two models.", body))
    s.append(Paragraph("2. Dataset", h))
    s.append(Paragraph(
        "<b>Posts:</b> Kaggle <i>gpreda/reddit-wallstreetsbets-posts</i> - 53,187 posts (Sep 2020 - Aug 2021) with title, score, id, url, "
        "comment count, creation time, body, timestamp. <b>Prices:</b> daily OHLC from Yahoo Finance (<i>yfinance</i>). "
        f"Posts mentioning each stock: GME {mentions['GME']:,}, AMC {mentions['AMC']:,}, TSLA {mentions['TSLA']:,}, FB {mentions['FB']:,} "
        "(ticker or company name). Each post becomes one example, as in the report.", body))
    s.append(Paragraph("3. Approach (replicating the report)", h))
    s.append(Paragraph(
        "<b>Pre-processing</b>: drop NaN titles, lowercase, remove URLs, handles, special and single characters, collapse spaces. "
        "<b>Features (7)</b>: post_score, total_comms_num, len_of_post, positive/neutral/negative sentiment (VADER), bias term; count "
        "features are log1p-transformed and standardised on the training set. "
        "<b>Labels</b>: <i>direction</i> = open(k+1) &gt; open(k); <i>volatility</i> = |open(k+1)/open(k)-1| above the training-set median "
        "(the report's 'significant shift'). Two timing conventions: <i>paper</i> (k = post's trading day) and <i>tradable</i> "
        "(k = first open after the post, so no look-ahead). "
        "<b>Models</b>: (1) logistic regression solved with <b>Newton's method implemented from scratch</b> (small L2 term keeps the Hessian "
        "invertible); (2) 3x256 ReLU network, sigmoid output, Adam (lr 1e-3), 30 epochs, batch 16 (64 for GME/TSLA), best epoch by validation "
        "accuracy; (3) uniform-random and majority-class baselines. "
        "<b>Tuning</b>: logistic-regression L2 in {1e-6..10} by validation log-loss; network random search (6 configs: depth/width, "
        "learning rate, weight decay, batch) by validation accuracy, re-trained with 5 seeds. "
        "<b>Evaluation</b>: chronological 80/10/10 train/val/test split, so the test set is strictly in the future.", body))
    g = VOL["GME"]
    s.append(Paragraph("4. New model: drift-aware shrinkage ensemble (our addition)", h))
    s.append(Paragraph(
        "<b>Why the report's models fail.</b> Training data is dominated by the squeeze month; the test months behave differently in two ways. "
        f"<i>Label shift</i>: 'volatile' means a move above the training median, so {100 * g['train_pos_rate']:.0f}% of GME training posts are volatile "
        f"but only {100 * g['test_pos_rate']:.0f}% of test posts. <i>Feature drift</i>: the subreddit grew, so GME's mean log post-score rises from "
        f"{g['mean_log_score']['train']:.1f} (train) to {g['mean_log_score']['test']:.1f} (test), about "
        f"{math.exp(g['mean_log_score']['test'] - g['mean_log_score']['train']):.0f}x more upvotes. Models that learned the squeeze keep "
        "predicting it and fall far below the majority baseline. The new model has four parts:", body))
    s.append(Paragraph(
        "<b>(a) Relative features.</b> Score, comments and length are replaced by their percentile rank among the stock's previous 200 posts "
        "(only earlier posts, so no look-ahead): 'top 10% of recent posts' means the same in February and in July. Sentiment is unchanged. "
        "<b>(b) Four models, soft vote.</b> Newton logistic regression, a random forest (bagging), gradient boosting (boosting) and Gaussian naive "
        "Bayes are trained on the same features; their probabilities are averaged with equal weights: p_vote. "
        "<b>(c) Shrinkage toward the prior.</b> The prior is the share of 1-labels in the training data, i.e. the best guess before reading the post "
        "(Bayes' prior). The final probability is p = &lambda; p_vote + (1-&lambda;) prior: &lambda; = 0 is the majority-class rule, &lambda; = 1 "
        "trusts the ensemble fully, so &lambda; sets the bias-variance trade-off. "
        "<b>(d) Tuning by forward-chaining cross-validation.</b> Random search over 8 configurations (L2; forest leaf size; boosting trees, depth, "
        f"learning rate), each with &lambda; in {{{', '.join(str(l) for l in ENS_LAMS)}}}: 48 candidates. Each is scored by log-loss on 4 time-ordered folds: "
        "fit on the oldest 40/60/80% of training and score the next 20%, then fit on all training and score validation (a fold whose training part "
        "holds only one class is skipped, e.g. AMC direction: 3 folds). Lowest mean log-loss wins (ties to the smaller &lambda;); the winner is refit on "
        "train+val and evaluated once on test. If the posts did not help predict the future in CV, &lambda; goes to 0 and the model falls back to the prior.", body))
    s.append(Image(FIGS["pipeline"], width=16.5 * cm, height=16.5 * cm * 4.2 / 12.5))
    s.append(Paragraph("5. Implementation overview", h))
    s.append(Paragraph(
        "Python; <i>src/wsb_pipeline.py</i> (data, features incl. relative ranks, labels, splits), <i>src/models.py</i> (Newton logistic regression, network, "
        "<i>ShrinkageEnsemble</i> and <i>tune_ensemble</i>), <i>src/run_experiments.py</i> (experiments, tuning, baselines, back-test, figures), "
        "<i>src/leakage_check.py</i> (shuffled vs chronological split), <i>src/demo.py</i> (live scoring of a post, and <i>--quick</i>, which runs the whole "
        "pipeline live on one stock), <i>tests/</i> (12 unit tests). Back-test: each trading day, majority vote of the day's per-post predictions "
        "-&gt; long (open to next open) or flat, compared with going long on every day that had posts.", body))
    s.append(Paragraph("6. Results (held-out chronological test set, accuracy)", h))
    s.append(Paragraph("<b>Direction task, 'tradable' labels</b>", body))
    s.append(tbl(table_data("direction", "tradable")))
    s.append(Spacer(1, 2))
    s.append(KeepTogether([Paragraph("<b>Volatility task, 'paper' labels</b>", body), tbl(table_data("volatility", "paper"))]))
    s.append(Spacer(1, 2))
    lk = leak.set_index(["stock", "split"])
    s.append(Paragraph(
        "<b>Leakage check</b> (report's direction task, report configuration): with a <i>shuffled</i> split the network scores "
        + ", ".join(f"{t} {lk.loc[(t, 'shuffled'), 'nn_acc']:.2f}" for t in T)
        + "; on our <i>chronological</i> split it scores "
        + ", ".join(f"{t} {lk.loc[(t, 'chronological'), 'nn_acc']:.2f}" for t in T)
        + ". Same-day posts share a label, so shuffling leaks the answer.", body))
    s.append(Paragraph(
        f"<b>New model over all {E['n']} settings</b> (both label modes, both tasks): it matches or beats the majority "
        f"baseline in <b>{E['match_maj']}/{E['n']}</b> (best tuned old model: {E['old_match_maj']}/{E['n']}) and matches or beats both tuned old models in "
        f"<b>{E['beat_old']}/{E['n']}</b>. Mean accuracy: ensemble {E['acc']['ens_acc']:.3f}, tuned LogReg {E['acc']['lr_tuned_acc']:.3f}, tuned NN "
        f"{E['acc']['nn_tuned_acc']:.3f}, majority {E['acc']['majority_acc']:.3f}. Mean test log-loss: ensemble {E['ll']['ens']:.3f}, LogReg "
        f"{E['ll']['lr_tuned']:.3f}, NN {E['ll']['nn_tuned']:.3f}, majority {E['ll']['majority']:.3f}. CV chose &lambda; = 0 (pure prior) in {E['lam0']} settings; "
        f"where &lambda; = 0 the model equals the majority baseline. Still below it: " + ", ".join(nice(k) for k in E["below"]) + ".", body))
    s.append(Image(os.path.join(RES, "fig_accuracy_volatility.png"), width=15.0 * cm, height=15.0 * cm * 3.8 / 10))
    s.append(Paragraph("7. Conclusions", h))
    best = {t: res[f"paper/volatility/{t}"] for t in T}
    s.append(Paragraph(
        "(i) <b>Honest evaluation matters.</b> We could not reproduce the report's high accuracies (GME 0.84, AMC 0.82) on a time-ordered split; a shuffled split inflates AMC and FB, which suggests leakage is a likely cause (we cannot verify the report's exact split). On a future "
        "hold-out the report's models do not beat the majority-class baseline on direction by more than the test-set standard error (AMC is far below it). "
        f"(ii) <b>Tuning the report's models helps only modestly:</b> AMC volatility network {best['AMC']['nn_acc']:.2f} to "
        f"{best['AMC']['nn_tuned_acc']:.2f} (only matching the majority baseline {best['AMC']['majority_acc']:.2f}); TSLA {best['TSLA']['nn_acc']:.2f} to "
        f"{best['TSLA']['nn_tuned_acc']:.2f} (still far below {best['TSLA']['majority_acc']:.2f}). "
        f"(iii) <b>The new shrinkage ensemble is the most reliable model</b>: majority-level in {E['match_maj']}/{E['n']} settings, with the largest gains where the old models "
        f"failed worst (GME volatility {gain('paper/volatility/GME')}, TSLA volatility {gain('paper/volatility/TSLA')}, AMC direction {gain('tradable/direction/AMC')}). "
        "It wins by <i>recognising when the posts do not generalise</i> and falling back to the prior; its test AUC stays near 0.5, so it finds no real predictive signal "
        "in the posts either. (iv) The back-tests show no edge: the network strategy stays flat on GME/AMC/TSLA, and neither strategy beats always-long. "
        "(v) Distribution shift (squeeze vs after) and tiny TSLA/FB test sets (124 and 26 posts, std. error 5-10 pp) limit what can be learned. "
        "<b>Future work:</b> richer text features (BERT embeddings), day-level aggregation, more data, and a transaction-cost-aware strategy.", body))
    doc.build(s)


# ------------------------------------------------------------------ slides
NAVY, TEAL, PINK, INK, MUTED = (RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x00, 0xA6, 0xA6), RGBColor(0xFB, 0xE3, 0xE7),
                                RGBColor(0x22, 0x22, 0x22), RGBColor(0x80, 0x80, 0x80))
OLD_KEYS = ["nn_acc", "nn_tuned_acc", "lr_acc", "lr_tuned_acc", "majority_acc", "random_acc"]
OLD_HEAD = ["Stock", "n test", "NN (report)", "NN (tuned)", "LogReg (report)", "LogReg (tuned)", "Majority", "Random"]


def build_pptx(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
    blank = prs.slide_layouts[6]
    count = [0]

    def text(s, txt, left, top, width, height, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, italic=False):
        tf = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text, p.alignment = txt, align
        p.font.size, p.font.bold, p.font.italic, p.font.color.rgb = Pt(size), bold, italic, color
        return tf

    def rect(s, left, top, width, height, fill):
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        r.fill.solid(); r.fill.fore_color.rgb = fill; r.line.fill.background()
        return r

    def new_slide(title, section):
        count[0] += 1
        s = prs.slides.add_slide(blank)
        rect(s, 0, 0, 13.33, 1.05, NAVY)
        rect(s, 0, 1.05, 13.33, 0.06, TEAL)
        text(s, title, 0.5, 0.2, 10.3, 0.7, 28, True, RGBColor(0xFF, 0xFF, 0xFF))
        text(s, section.upper(), 10.4, 0.36, 2.5, 0.4, 12, True, TEAL, PP_ALIGN.RIGHT)
        text(s, "WSB market prediction  |  UE24CS352A Machine Learning", 0.5, 7.05, 8, 0.35, 10, color=MUTED)
        text(s, str(count[0]), 12.2, 7.05, 0.6, 0.35, 10, color=MUTED, align=PP_ALIGN.RIGHT)
        return s

    def bullets(s, items, left=0.6, top=1.45, width=12.1, size=18):
        tf = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(1)).text_frame
        tf.word_wrap = True
        for i, b in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = "▪  " + b
            run.font.size, run.font.color.rgb = Pt(size), INK
            p.space_after = Pt(9)
        chars = width * 72 / (size * 0.5)
        return top + sum(math.ceil((len(b) + 3) / chars) * size * 1.22 / 72 + 9 / 72 for b in items)

    def picture(s, img, left, top, max_w, max_h):
        w, h = PILImage.open(img).size
        sc = min(max_w / w, max_h / h)
        s.shapes.add_picture(img, Inches(left + (max_w - w * sc) / 2), Inches(top), width=Inches(w * sc))

    def table(s, data, top, size=14, row_h=0.4, highlight=None, left=0.6, width=12.1):
        rows, cols = len(data), len(data[0])
        g = s.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(row_h * rows)).table
        for i, r_ in enumerate(data):
            for j, v in enumerate(r_):
                c = g.cell(i, j)
                c.text = str(v).replace("\n", " ")
                para = c.text_frame.paragraphs[0]
                para.font.size = Pt(size)
                para.font.bold = i == 0 or (highlight is not None and j == highlight)
                c.fill.solid()
                if i == 0:
                    c.fill.fore_color.rgb = NAVY
                    para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                else:
                    c.fill.fore_color.rgb = PINK if j == highlight else (RGBColor(0xF2, 0xF5, 0xF8) if i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
                    para.font.color.rgb = INK
        return top + row_h * rows

    def note(s, txt):
        text(s, txt, 0.6, 6.5, 12.1, 0.5, 14, italic=True, color=RGBColor(0x55, 0x55, 0x55))

    # ---- 1. title
    count[0] += 1
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 13.33, 7.5, NAVY)
    rect(s, 1.0, 4.05, 3.0, 0.07, TEAL)
    text(s, "Predicting Market Volatility & Short-Term Trading Signals from r/WallStreetBets",
         1.0, 1.9, 11.3, 2.0, 38, True, RGBColor(0xFF, 0xFF, 0xFF))
    text(s, "Replicating a CS229 report, testing it honestly, and building a better model",
         1.0, 4.3, 11.3, 0.6, 20, color=RGBColor(0xCF, 0xE8, 0xE8))
    text(s, "UE24CS352A Machine Learning - Mini-Project\nPrithvi Aithal (PES1UG24CS909)  |  Sagar A Walikar (PES1UG24CS396)",
         1.0, 5.3, 11.3, 1.0, 16, color=RGBColor(0xFF, 0xFF, 0xFF))

    # ---- 2. agenda
    s = new_slide("Agenda", "Overview")
    agenda = [("1", "Problem & data", "what we predict and from what"),
              ("2", "Replicating the report", "Newton logistic regression and a neural network"),
              ("3", "Why it fails", "data leakage and the market changing after the squeeze"),
              ("4", "Our new model", "a shrinkage ensemble, tuned with time-ordered cross-validation"),
              ("5", "Results", "all models vs baselines, and a trading back-test"),
              ("6", "Wrap-up", "conclusions and live demo")]
    for i, (n_, head, sub) in enumerate(agenda):
        y = 1.55 + i * 0.88
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.9), Inches(y), Inches(0.6), Inches(0.6))
        circ.fill.solid(); circ.fill.fore_color.rgb = TEAL if n_ == "4" else NAVY; circ.line.fill.background()
        circ.text_frame.paragraphs[0].text = n_
        circ.text_frame.paragraphs[0].font.size, circ.text_frame.paragraphs[0].font.bold = Pt(18), True
        circ.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        text(s, head, 1.8, y - 0.02, 4.5, 0.5, 22, True)
        text(s, sub, 5.6, y + 0.04, 7.2, 0.5, 18, color=RGBColor(0x55, 0x55, 0x55))

    # ---- section 1: problem & data
    s = new_slide("Problem statement", "Problem & data")
    bullets(s, ["r/WallStreetBets (WSB) retail traders drove the January 2021 GameStop short squeeze",
                "Question: do the posts themselves carry a usable signal about the next day?",
                "Task 1 - direction: will the price be higher at the next open?",
                "Task 2 - volatility: will the next move be unusually large?",
                "Task 3 - trading: can the predictions drive a simple long / flat strategy?",
                "Plan: replicate the CS229 report -> test it honestly -> build a better model"])
    s = new_slide("Dataset", "Problem & data")
    bullets(s, ["Kaggle gpreda/reddit-wallstreetsbets-posts: 53,187 posts, Sep 2020 - Aug 2021",
                "Daily prices from Yahoo Finance (yfinance)",
                f"Posts mentioning: GME {mentions['GME']:,} | AMC {mentions['AMC']:,} | TSLA {mentions['TSLA']:,} | FB {mentions['FB']:,}",
                "One example per post that mentions the stock"], width=6.0)
    picture(s, os.path.join(RES, "fig_mentions.png"), 6.9, 1.5, 6.0, 5.2)
    s = new_slide("Features, labels and evaluation", "Problem & data")
    bullets(s, ["Cleaning: lowercase; remove URLs, user handles, special and single characters",
                "7 features per post: score, comments, length, VADER positive / neutral / negative sentiment, bias",
                "Direction label: open(k+1) > open(k)",
                "Volatility label: |next move| larger than the training-period median move",
                "Two timings: 'paper' (as in the report) and 'tradable' (only the move after the post - no look-ahead)",
                "Chronological 80 / 10 / 10 train / validation / test split: the test set is strictly in the future",
                "Baselines: random guess, and 'always predict the majority class' - the bar every model must clear"])

    # ---- section 2: replication
    s = new_slide("The report's models", "Replicating the report")
    bullets(s, ["Logistic regression solved with Newton's method, written from scratch (small L2 keeps the Hessian invertible)",
                "Neural network: 3 x 256 ReLU layers, sigmoid output, Adam (lr 1e-3), 30 epochs, best epoch on validation",
                "Our tuning: L2 strength by grid search (validation log-loss); network by random search over depth / width, "
                "learning rate, weight decay, batch size (validation accuracy); 5 seeds averaged"])
    s = new_slide("Newton's method converges in 4-8 iterations", "Replicating the report")
    bullets(s, ["Each step solves H · Δw = gradient of the logistic loss",
                "||Δw|| falls below 1e-6 in 4-8 iterations for every stock",
                "Unlike the report, AMC also converges"], width=5.6)
    picture(s, os.path.join(RES, "fig_convergence.png"), 6.4, 1.5, 6.5, 5.2)
    s = new_slide("Results of the report's models (test accuracy)", "Replicating the report")
    text(s, "Direction  ('tradable' labels)", 0.6, 1.3, 8, 0.4, 16, True)
    table(s, [OLD_HEAD] + [row("direction", "tradable", t, OLD_KEYS) for t in T], 1.72, 13, 0.36)
    text(s, "Volatility  ('paper' labels)", 0.6, 3.95, 8, 0.4, 16, True)
    table(s, [OLD_HEAD] + [row("volatility", "paper", t, OLD_KEYS) for t in T], 4.37, 13, 0.36)
    gv, ad = res["paper/volatility/GME"], res["tradable/direction/AMC"]
    note(s, f"Almost nothing beats the Majority column: e.g. GME volatility {max(gv[o] for o in OLD):.2f} vs {gv['majority_acc']:.2f}, "
            f"AMC direction {max(ad[o] for o in OLD):.2f} vs {ad['majority_acc']:.2f}.")

    # ---- section 3: why it fails
    lk = leak.set_index(["stock", "split"])
    s = new_slide("Why we cannot match the report: leakage", "Why it fails")
    bullets(s, ["Posts from the same day share one label",
                "A shuffled split puts near-copies of test posts in training -> the model memorises, not predicts",
                "Same network, same data, only the split changes:"])
    table(s, [["Stock", "Shuffled split", "Chronological split (ours)"]] +
          [[t, f"{lk.loc[(t, 'shuffled'), 'nn_acc']:.2f}", f"{lk.loc[(t, 'chronological'), 'nn_acc']:.2f}"] for t in T],
          3.35, 16, 0.45, left=2.0, width=9.3)
    note(s, "The report does not state its split, so leakage is a likely - not proven - reason for its 0.84.")
    g = VOL["GME"]
    s = new_slide("Why the models fail on future data: the market changed", "Why it fails")
    y = bullets(s, [f"Label shift: ~{100 * g['train_pos_rate']:.0f}% of GME training posts are 'volatile', only "
                    f"{100 * g['test_pos_rate']:.0f}% of test posts - the market calmed down after the squeeze",
                    f"Feature drift: WSB grew, GME mean log-score {g['mean_log_score']['train']:.1f} -> {g['mean_log_score']['test']:.1f} "
                    f"(~{math.exp(g['mean_log_score']['test'] - g['mean_log_score']['train']):.0f}x more upvotes)",
                    "Models trained on the squeeze keep predicting the squeeze -> below 'always guess the majority'"], size=16)
    picture(s, FIGS["drift"], 0.6, y + 0.05, 12.1, 6.95 - y - 0.05)

    # ---- section 4: new model
    s = new_slide("Our new model at a glance: shrinkage ensemble", "Our new model")
    y = bullets(s, ["Four syllabus models vote; the vote is pulled toward the majority answer by an amount λ",
                    "λ and the models' settings are learned from past -> future cross-validation, never from the test set"], size=17)
    picture(s, FIGS["pipeline"], 0.6, y + 0.1, 12.1, 6.95 - y - 0.1)
    s = new_slide("Piece 1: relative features (fixes feature drift)", "Our new model")
    bullets(s, ["Score, comments and length -> percentile rank among the stock's previous 200 posts",
                "500 upvotes was huge in February and ordinary in July; 'top 10% of recent posts' means the same in any month",
                "Only earlier posts are used, so there is no look-ahead",
                "Sentiment (VADER) and the bias column are unchanged",
                f"Biggest effect where drift is worst: GME volatility, new model vs old models {gain('paper/volatility/GME')}"])
    ex = [["Model", "What it brings", "Its probability"]]
    for (name, p), why in zip(EX_PARTS, ["linear boundary, stable (our Newton solver)",
                                         "many trees on random samples, averaged (bagging)",
                                         "small trees added one by one, each fixing the last (boosting)",
                                         "probabilistic, uses Bayes' theorem"]):
        ex.append([name, why, f"{p:.2f}"])
    ex.append(["Ensemble  =  p_vote", "simple average, equal weights (soft voting)", f"{EX_VOTE:.2f}"])
    s = new_slide("Piece 2: four models, soft voting", "Our new model")
    y = bullets(s, [f"Worked example - the demo post \"{EX_POST['title']}\" ({EX_POST['score']} upvotes, {EX_POST['comments']} comments), GME direction",
                    "Soft voting averages probabilities (hard voting would count 0/1 votes); different models make different mistakes"], size=16)
    table(s, ex, y + 0.1, 15, 0.48)
    shrink = [["λ", "p  =  λ · p_vote  +  (1 − λ) · prior", "Prediction"]]
    for lam in (1.0, 0.5, 0.25, 0.0):
        p = lam * EX_VOTE + (1 - lam) * EX_PRIOR
        shrink.append([f"{lam}", f"{lam} × {EX_VOTE:.2f} + {1 - lam:.2f} × {EX_PRIOR:.2f} = {p:.2f}",
                       "up (buy)" if p >= .5 else "down (stay flat)"])
    s = new_slide("Piece 3: the prior and λ (shrinkage)", "Our new model")
    y = bullets(s, [f"Prior = share of 1-labels in the training data = the best guess before reading the post (Bayes' prior). "
                    f"GME direction: {EX_PRIOR:.2f} - the price rose after {100 * EX_PRIOR:.0f}% of posts",
                    "λ = 0 -> p = prior for every post = the majority baseline;   λ = 1 -> fully trust the 4 models",
                    "Small λ: more bias, less variance (not fooled by noise).  Large λ: flexible, but can chase noise",
                    f"Same post with different λ (for this model, cross-validation chose λ = {EX_LAM}):"], size=16)
    table(s, shrink, y + 0.05, 15, 0.45)
    folds_txt = (f"4 folds ({', '.join(E['short_folds'])}: 3 - its oldest window has only one class)" if E["short_folds"] else "4 folds")
    s = new_slide("Piece 4: tuning with forward-chaining cross-validation", "Our new model")
    y = bullets(s, ["Random search: 8 settings (L2; forest leaf size; boosting trees, depth, learning rate) x 6 λ values = 48 candidates",
                    f"{folds_txt}: always fit on the past, score on the future - k-fold would shuffle time and leak",
                    "Score = log-loss; lowest average wins (ties -> smaller λ); the winner is refit on train + val; test used once"], size=16)
    picture(s, FIGS["cv"], 0.6, y + 0.05, 12.1, 6.95 - y - 0.05)
    lam_counts = ",  ".join(f"λ={lam}: {c}" for lam, c in E["lams"].items() if c)
    s = new_slide("How λ was actually chosen", "Our new model")
    bullets(s, ["λ needs no retraining: each fold's p_vote is re-mixed with the prior for every λ and scored",
                "Curve falls -> the posts helped on past -> future folds -> large λ",
                "Curve rises -> the posts misled the models -> λ = 0, fall back to the majority answer",
                f"Chosen over the 16 settings:  {lam_counts}"], width=5.3, size=17)
    picture(s, FIGS["lambda"], 6.1, 1.4, 6.9, 5.4)

    # ---- section 5: results
    s = new_slide("All models vs baselines (test accuracy)", "Results")
    text(s, "Direction  ('tradable' labels)", 0.6, 1.3, 8, 0.4, 16, True)
    head = [h.replace("\n", " ") for h in HEAD]
    table(s, [head] + [row("direction", "tradable", t, KEYS) for t in T], 1.72, 13, 0.36, highlight=6)
    text(s, "Volatility  ('paper' labels)", 0.6, 3.95, 8, 0.4, 16, True)
    table(s, [head] + [row("volatility", "paper", t, KEYS) for t in T], 4.37, 13, 0.36, highlight=6)
    note(s, "Highlighted = our new model. Compare every model with Majority, not with 0.5.")
    s = new_slide("New model vs the report's models: all 16 settings", "Results")
    summary = [["", "Shrinkage ensemble (new)", "LogReg (tuned)", "NN (tuned)", "Majority"],
               ["Settings at or above majority", f"{E['match_maj']} / {E['n']}",
                f"{sum(res[k]['lr_tuned_acc'] >= res[k]['majority_acc'] - 1e-9 for k in SETTINGS)} / {E['n']}",
                f"{sum(res[k]['nn_tuned_acc'] >= res[k]['majority_acc'] - 1e-9 for k in SETTINGS)} / {E['n']}", "-"],
               ["Mean accuracy", f"{E['acc']['ens_acc']:.3f}", f"{E['acc']['lr_tuned_acc']:.3f}", f"{E['acc']['nn_tuned_acc']:.3f}",
                f"{E['acc']['majority_acc']:.3f}"],
               ["Mean log-loss (lower = better)", f"{E['ll']['ens']:.3f}", f"{E['ll']['lr_tuned']:.3f}", f"{E['ll']['nn_tuned']:.3f}",
                f"{E['ll']['majority']:.3f}"]]
    y = table(s, summary, 1.45, 15, 0.5, highlight=1)
    bullets(s, [f"Matches or beats both old models in {E['beat_old']}/{E['n']} settings; biggest gains: GME volatility {gain('paper/volatility/GME')}, "
                f"TSLA volatility {gain('paper/volatility/TSLA')}, AMC direction {gain('tradable/direction/AMC')}",
                "Still below majority: " + "; ".join(nice(k) for k in E["below"]),
                "Test AUC ~ 0.5: it wins by not being fooled, not by finding a real signal"], top=y + 0.3, size=16)
    b = {r["stock"]: r for r in bt}
    s = new_slide("Trading back-test", "Results")
    y = bullets(s, ["Each day: majority vote of that day's post predictions -> long (open to next open) or flat",
                    "Network strategy: never goes long on GME / AMC / TSLA; no edge on FB",
                    f"Ensemble strategy: FB {b['FB']['ensemble_return_%']:+.1f}% (always-long {b['FB']['always_long_return_%']:+.1f}%); AMC "
                    f"{b['AMC']['ensemble_return_%']:+.1f}% on {b['AMC']['ensemble_days_traded']} days vs always-long {b['AMC']['always_long_return_%']:+.0f}%",
                    "No strategy shows a real edge"], size=17)
    picture(s, os.path.join(RES, "fig_backtest.png"), 0.6, y + 0.1, 12.1, 6.95 - y - 0.1)

    # ---- section 6: wrap-up
    s = new_slide("Conclusions", "Wrap-up")
    bullets(s, ["Honest, time-ordered evaluation: the report's accuracies do not reproduce; shuffled splits likely leak",
                "The report's models, even tuned, mostly fall below 'always guess the majority'",
                "Cause: the market and the subreddit changed after the squeeze (label shift + feature drift)",
                f"Our shrinkage ensemble handles this best: majority-level in {E['match_maj']}/{E['n']} settings (old: {E['old_match_maj']}/{E['n']}), "
                "best mean accuracy and log-loss",
                "It wins by knowing when not to trust the posts - WSB post features carry no reliable next-day signal",
                "Future work: BERT text features, day-level aggregation, more data, cost-aware strategy"])
    s = new_slide("Live demo", "Wrap-up")
    table(s, [["Command", "What it shows"],
              ["python src/demo.py --quick TSLA", "the whole pipeline live: posts -> features -> Newton LR + network + tuned ensemble "
                                                  "(prints the CV score for every λ) -> accuracy vs baselines (~50 s)"],
              ["python src/demo.py --stock GME --title \"GME to the moon, diamond hands\" --score 1200 --comments 340",
               "scores a typed post with all three saved models (the worked example on slide 13)"],
              ["python src/demo.py --stock TSLA --task volatility --title \"Tesla earnings tomorrow\"", "the volatility task"],
              ["python src/demo.py --replay GME", "back-test summary for one stock"]], 1.5, 15, 0.7)
    note(s, "run_experiments.py regenerates all of results/ and models/ (about 10-15 minutes, so not run live).")
    count[0] += 1
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 13.33, 7.5, NAVY)
    text(s, "Thank you", 1.0, 2.5, 11.3, 1.2, 48, True, RGBColor(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER)
    text(s, "Questions?", 1.0, 3.7, 11.3, 0.8, 26, color=RGBColor(0xCF, 0xE8, 0xE8), align=PP_ALIGN.CENTER)
    text(s, "github.com/Prithvi-Aithal/wsb-market-prediction", 1.0, 5.2, 11.3, 0.5, 16,
         color=RGBColor(0xCF, 0xE8, 0xE8), align=PP_ALIGN.CENTER)
    prs.core_properties.author = prs.core_properties.last_modified_by = "Prithvi Aithal, Sagar A Walikar"
    prs.core_properties.title = "WSB market prediction - mini-project"
    prs.save(path)


if __name__ == "__main__":
    draw_figures()
    build_pdf(os.path.join(HERE, "writeup.pdf"))
    build_pptx(os.path.join(HERE, "presentation.pptx"))
    print("built docs/writeup.pdf, docs/presentation.pptx and docs/fig_*.png")
