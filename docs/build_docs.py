"""Build the 2-page write-up (PDF) and the presentation (PPTX) from results/.

    python docs/build_docs.py
"""
import json
import os

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
R = json.load(open(os.path.join(RES, "results.json")))
res, mentions, bt = R["results"], R["mentions"], R["backtest"]
leak = pd.read_csv(os.path.join(RES, "leakage_check.csv"))
T = ["GME", "AMC", "TSLA", "FB"]


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
}


def gain(key):
    r = res[key]
    return f"{r['ens_acc']:.2f} vs {max(r[o] for o in OLD):.2f}"


# ------------------------------------------------------------------ PDF (2 pages)
def build_pdf(path):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=10.6, spaceAfter=2.5)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=10, spaceBefore=4, spaceAfter=2)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=14, spaceAfter=2)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="Mini-project write-up: WSB market prediction",
                            author="Prithvi Aithal, Sagar A Walikar")
    s = []

    def tbl(data):
        t = Table(data, hAlign="CENTER", colWidths=[1.3 * cm, 1.3 * cm] + [1.85 * cm] * 7)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9f2f2")),
                               ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
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
        "(a drift-aware shrinkage ensemble) to improve on the report's two models.", body))
    s.append(Paragraph("2. Dataset", h))
    s.append(Paragraph(
        "<b>Posts:</b> Kaggle <i>gpreda/reddit-wallstreetsbets-posts</i> - 53,187 posts (Sep 2020 - Aug 2021) with title, score, id, url, "
        "comment count, creation time, body, timestamp. <b>Prices:</b> daily OHLC from Yahoo Finance (<i>yfinance</i>). "
        f"Posts mentioning each stock: GME {mentions['GME']:,}, AMC {mentions['AMC']:,}, TSLA {mentions['TSLA']:,}, FB {mentions['FB']:,} "
        "(ticker or company name). Each post becomes one example, as in the report.", body))
    s.append(Paragraph("3. Approach", h))
    s.append(Paragraph(
        "<b>Pre-processing</b>: drop NaN titles, lowercase, remove URLs, handles, special and single characters, collapse spaces. "
        "<b>Features (7)</b>: post_score, total_comms_num, len_of_post, positive/neutral/negative sentiment (VADER), bias term; count "
        "features are log1p-transformed and standardised on the training set. "
        "<b>Labels</b>: <i>direction</i> = open(k+1) &gt; open(k); <i>volatility</i> = |open(k+1)/open(k)-1| above the training-set median "
        "(the report's 'significant shift'). Two timing conventions: <i>paper</i> (k = post's trading day) and <i>tradable</i> "
        "(k = first open after the post, so no look-ahead). "
        "<b>Models</b>: (1) logistic regression solved with <b>Newton's method implemented from scratch</b> (small L2 term keeps the Hessian "
        "invertible); (2) 3x256 ReLU network, sigmoid output, Adam (lr 1e-3), 30 epochs, batch 16 (64 for GME/TSLA), best epoch by validation "
        "accuracy; (3) uniform-random and majority-class baselines; (4) <b>new: shrinkage ensemble</b> - soft vote of Newton logistic regression, "
        "random forest, gradient boosting and Gaussian naive Bayes, blended with the training base rate: p = &lambda; p<sub>vote</sub> + (1-&lambda;) prior. "
        "Its count features are <i>relative</i> (percentile rank of score/comments/length among the stock's last 200 posts, only past posts used), "
        "because raw engagement drifts about 10x as the subreddit grew after the squeeze. &lambda; (0 = majority rule, 1 = plain ensemble) and the base "
        "learners' hyper-parameters (random search, 8 configs: L2, forest leaf size, boosting trees/depth/learning rate) are chosen by "
        "<b>forward-chaining cross-validation</b> log-loss on train+val; the chosen model is refit on train+val. "
        "<b>Tuning</b>: logistic-regression L2 in {1e-6..10} by validation log-loss; network random search (6 configs: depth/width, "
        "learning rate, weight decay, batch) by validation accuracy, re-trained with 5 seeds. "
        "<b>Evaluation</b>: chronological 80/10/10 train/val/test split so the test set is strictly in the future.", body))
    s.append(Paragraph("4. Implementation overview", h))
    s.append(Paragraph(
        "Python; <i>src/wsb_pipeline.py</i> (data, features, labels, splits), <i>src/models.py</i> (Newton logistic regression, network), "
        "<i>src/run_experiments.py</i> (experiments, tuning, baselines, back-test, figures), <i>src/leakage_check.py</i> (shuffled vs chronological split), "
        "<i>src/demo.py</i> (live scoring of a post, and a <i>--quick</i> mode that runs the whole pipeline live on one stock), <i>tests/</i> (unit tests). Back-test: for every trading day, majority vote of the day's per-post predictions "
        "-&gt; long (open to next open) or flat, compared with going long on every day that had posts.", body))
    s.append(Paragraph("5. Results (held-out chronological test set, accuracy)", h))
    s.append(Paragraph("<b>Direction task, 'tradable' labels</b>", body))
    s.append(tbl(table_data("direction", "tradable")))
    s.append(Spacer(1, 3))
    s.append(Paragraph("<b>Volatility task, 'paper' labels</b>", body))
    s.append(tbl(table_data("volatility", "paper")))
    s.append(Spacer(1, 3))
    lk = leak.set_index(["stock", "split"])
    s.append(Paragraph(
        "<b>Leakage check</b> (report's direction task, report configuration): with a <i>shuffled</i> split the network scores "
        + ", ".join(f"{t} {lk.loc[(t, 'shuffled'), 'nn_acc']:.2f}" for t in T)
        + "; on our <i>chronological</i> split it scores "
        + ", ".join(f"{t} {lk.loc[(t, 'chronological'), 'nn_acc']:.2f}" for t in T)
        + ". Same-day posts share a label, so shuffling leaks the answer.", body))
    s.append(Paragraph(
        f"<b>New model over all {E['n']} settings</b> (both label modes, both tasks): the shrinkage ensemble matches or beats the majority "
        f"baseline in <b>{E['match_maj']}/{E['n']}</b> (best tuned old model: {E['old_match_maj']}/{E['n']}) and matches or beats both tuned old models in "
        f"<b>{E['beat_old']}/{E['n']}</b>. Mean accuracy: ensemble {E['acc']['ens_acc']:.3f}, tuned LogReg {E['acc']['lr_tuned_acc']:.3f}, tuned NN "
        f"{E['acc']['nn_tuned_acc']:.3f}, majority {E['acc']['majority_acc']:.3f}. Mean test log-loss: ensemble {E['ll']['ens']:.3f}, LogReg "
        f"{E['ll']['lr_tuned']:.3f}, NN {E['ll']['nn_tuned']:.3f}, majority {E['ll']['majority']:.3f}. CV set &lambda; = 0 (pure prior) in {E['lam0']} settings.", body))
    s.append(Image(os.path.join(RES, "fig_accuracy_volatility.png"), width=15.5 * cm, height=15.5 * cm * 3.8 / 10))
    s.append(Paragraph("6. Conclusions", h))
    best = {t: res[f"paper/volatility/{t}"] for t in T}
    s.append(Paragraph(
        "(i) <b>Honest evaluation matters.</b> We could not reproduce the report's high accuracies (GME 0.84, AMC 0.82) on a time-ordered split; a shuffled split inflates AMC and FB, which suggests leakage is a likely cause (we cannot verify the report's exact split). On a future "
        "hold-out no direction model beats the majority-class baseline by more than the test-set standard error (see table; AMC is far below it), so we cannot claim WSB post "
        "features predict next-day direction from these 7 features. "
        f"(ii) <b>Tuning helps only modestly:</b> for AMC volatility the tuned network improves from {best['AMC']['nn_acc']:.2f} to "
        f"{best['AMC']['nn_tuned_acc']:.2f} (only matching the trivial majority baseline {best['AMC']['majority_acc']:.2f}), and for TSLA from "
        f"{best['TSLA']['nn_acc']:.2f} to {best['TSLA']['nn_tuned_acc']:.2f} (still far below its majority baseline {best['TSLA']['majority_acc']:.2f}); for the rest it mostly selects the trivial 'predict the majority' solution. "
        f"(iii) <b>The new shrinkage ensemble is the most reliable model</b>: it reaches the majority baseline in {E['match_maj']}/{E['n']} settings and lifts the "
        f"hardest cases most (GME volatility {gain('paper/volatility/GME')}, AMC direction {gain('tradable/direction/AMC')}, "
        f"TSLA volatility {gain('paper/volatility/TSLA')}). Its strength is knowing when the features "
        "do <i>not</i> generalise (it then falls back to the prior); its AUC stays near 0.5, so it still finds no real predictive signal in the posts. "
        "(iv) The back-test confirms it: the network strategy stays flat on GME/AMC/TSLA and slightly under-performs always-long on FB. "
        "(v) Distribution shift is the main obstacle: post volume peaks during the Jan-Feb 2021 squeeze, so train and test regimes differ; "
        "TSLA and FB test sets are tiny (124 and 26 posts, std. error 5-10 pp), so their differences are not significant. "
        "<b>Future work:</b> richer text features (BERT embeddings), day-level aggregation, more data, and a transaction-cost-aware strategy.", body))
    doc.build(s)


# ------------------------------------------------------------------ slides
def build_pptx(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)

    def slide(title, bullets=None, img=None, table=None):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        s.shapes.title.text_frame.paragraphs[0].font.size = Pt(30)
        # layouts are sized for a 10 in slide; place the title explicitly (all four values) on the 13.33 in slide
        t = s.shapes.title
        t.left, t.top, t.width, t.height = Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.9)
        top = 1.4
        if bullets:
            w = 6.2 if img else 12.3
            tb = s.shapes.add_textbox(Inches(0.5), Inches(top), Inches(w), Inches(5.5)).text_frame
            tb.word_wrap = True
            for i, b in enumerate(bullets):
                p = tb.paragraphs[0] if i == 0 else tb.add_paragraph()
                p.text = "- " + b
                p.font.size = Pt(18)
                p.space_after = Pt(8)
        if img:
            x = 7.0 if bullets else 0.7
            s.shapes.add_picture(img, Inches(x), Inches(top), width=Inches(6.0 if bullets else 12))
        if table:
            rows, cols = len(table), len(table[0])
            g = s.shapes.add_table(rows, cols, Inches(0.5), Inches(top), Inches(12.3), Inches(0.5 * rows)).table
            for i, r_ in enumerate(table):
                for j, v in enumerate(r_):
                    g.cell(i, j).text = v.replace("\n", " ")
                    g.cell(i, j).text_frame.paragraphs[0].font.size = Pt(14)

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Predicting Market Volatility & Short-Term Trading Signals from r/WallStreetBets"
    s.shapes.title.text_frame.paragraphs[0].font.size = Pt(36)
    t, sub_ = s.shapes.title, s.placeholders[1]
    t.left, t.top, t.width, t.height = Inches(1.0), Inches(2.0), Inches(11.33), Inches(1.9)
    sub_.left, sub_.top, sub_.width, sub_.height = Inches(1.0), Inches(4.2), Inches(11.33), Inches(1.4)
    s.placeholders[1].text = "UE24CS352A Machine Learning - Mini-Project\nPrithvi Aithal (PES1UG24CS909) | Sagar A Walikar (PES1UG24CS396)"
    slide("Problem statement", [
        "WSB drove the Jan-2021 GameStop short squeeze",
        "Can post features predict next-day price direction and volatility?",
        "Can the predictions drive a simple short-term trading strategy?",
        "Replicate the CS229 report, then extend it: tuning, strict baselines, leakage-free evaluation, back-test"])
    slide("Dataset", [
        "Kaggle gpreda/reddit-wallstreetsbets-posts: 53,187 posts, Sep 2020 - Aug 2021",
        "Prices: Yahoo Finance daily OHLC (yfinance)",
        f"Mentions: GME {mentions['GME']:,} | AMC {mentions['AMC']:,} | TSLA {mentions['TSLA']:,} | FB {mentions['FB']:,}",
        "One example per post that mentions the stock"], img=os.path.join(RES, "fig_mentions.png"))
    slide("Features and labels", [
        "7 features: score, comments, length, VADER pos/neu/neg, bias",
        "Preprocessing: lowercase, strip URLs/handles/special chars, log1p + standardise",
        "Direction: open(k+1) > open(k)",
        "Volatility: |move| > training median",
        "'paper' vs 'tradable' timing (no look-ahead)",
        "Chronological 80/10/10 split"])
    slide("Models", [
        "Logistic regression - Newton's method written from scratch (+ small L2)",
        "Neural net: 3x256 ReLU, sigmoid head, Adam 1e-3, 30 epochs",
        "Baselines: random guess, majority class",
        "Tuning: L2 grid (val log-loss); NN random search over depth/width, lr, weight decay, batch (val accuracy)",
        "NN results averaged over 5 seeds",
        "NEW: shrinkage ensemble (next slides)"])
    slide("New model: shrinkage ensemble", [
        "Soft vote of 4 learners: Newton LogReg + Random Forest (bagging) + Gradient Boosting + Gaussian Naive Bayes",
        "Shrunk toward the class prior: p = lam * p_vote + (1 - lam) * training base rate",
        "lam = 0 -> majority rule, lam = 1 -> plain ensemble: a bias-variance trade-off picked from data",
        "Drift-robust features: score / comments / length as percentile rank among the stock's last 200 posts (past only)",
        "Why: raw post scores grow ~10x after the squeeze (GME mean log-score 3.2 train -> 5.8 test)"])
    slide("Tuning the new model", [
        "Random search: 8 configs (L2; forest min leaf; boosting trees, depth, learning rate) x lam grid {0 ... 1}",
        "Forward-chaining CV: fit on first 40/60/80% of train, score next 20%; fit on train, score val",
        "Selection by log-loss (a proper probabilistic score), ties -> smaller lam (simpler model)",
        "Winner refit on train + val (most recent labelled data); test set never touched",
        f"CV chose lam = 0 (trust the prior) in {E['lam0']} of {E['n']} settings: post features mostly do not generalise forward"])
    slide("Newton convergence", ["||delta w|| falls below 1e-6 in 4-8 iterations for every stock",
                                 "Unlike the report, AMC also converges (6 iterations in the plotted run, 8 at most)"],
          img=os.path.join(RES, "fig_convergence.png"))
    slide("Results: direction (tradable labels)", table=table_data("direction", "tradable"))
    slide("Results: volatility (paper labels)", table=table_data("volatility", "paper"))
    slide("Accuracy overview", img=os.path.join(RES, "fig_accuracy_volatility.png"))
    slide("New model vs the report's models (all 16 settings)", [
        f"Matches or beats the majority baseline: ensemble {E['match_maj']}/{E['n']} vs best tuned old model {E['old_match_maj']}/{E['n']}",
        f"Matches or beats both tuned old models in {E['beat_old']}/{E['n']} settings",
        f"Mean accuracy: ensemble {E['acc']['ens_acc']:.3f} | LogReg {E['acc']['lr_tuned_acc']:.3f} | NN {E['acc']['nn_tuned_acc']:.3f} | majority {E['acc']['majority_acc']:.3f}",
        f"Mean test log-loss (lower = better): ensemble {E['ll']['ens']:.3f} | LogReg {E['ll']['lr_tuned']:.3f} | NN {E['ll']['nn_tuned']:.3f} | majority {E['ll']['majority']:.3f}",
        f"Biggest gains: GME volatility {gain('paper/volatility/GME')}; TSLA volatility {gain('paper/volatility/TSLA')}; AMC direction {gain('tradable/direction/AMC')}",
        "Below majority on AMC direction, GME volatility (regime shift) and FB paper-direction (26 posts); AUC ~ 0.5: no real signal"])
    lk = leak.set_index(["stock", "split"])
    slide("Why we do not match the report", [
        "Same-day posts share a label -> shuffled splits can leak; the report's exact split is not stated, so this is a likely, not proven, cause",
        "NN accuracy shuffled vs chronological: " + "; ".join(
            f"{t} {lk.loc[(t, 'shuffled'), 'nn_acc']:.2f} vs {lk.loc[(t, 'chronological'), 'nn_acc']:.2f}" for t in T),
        "Other differences: data to Aug 2021, our own sentiment/label details. Our chronological split is the honest estimate"])
    b = {r["stock"]: r for r in bt}
    slide("Trading back-test", ["Day-level majority vote of post predictions -> long or flat",
                                "NN strategy stays flat for GME/AMC/TSLA (never predicts 'up'); no edge on FB (13 trades)",
                                f"Ensemble strategy: FB {b['FB']['ensemble_return_%']:+.1f}% (= always-long), AMC {b['AMC']['ensemble_return_%']:+.1f}% "
                                f"on {b['AMC']['ensemble_days_traded']} days vs always-long {b['AMC']['always_long_return_%']:+.0f}%; flat on GME/TSLA",
                                "No strategy shows a real edge"], img=os.path.join(RES, "fig_backtest.png"))
    slide("Conclusions", [
        "We could not reproduce the report's accuracies; honest (time-ordered) evaluation shows no significant direction edge",
        "Tuning the report's models helps only modestly",
        f"Our new shrinkage ensemble is the most reliable: majority-level in {E['match_maj']}/{E['n']} settings, best mean accuracy and log-loss",
        "It wins by knowing when not to trust the posts, so it still finds no real predictive edge",
        "Distribution shift (squeeze vs after) and small samples limit what can be learned",
        "Future: BERT features, daily aggregation, more data, cost-aware strategy"])
    slide("Live demo", ["python src/demo.py --quick TSLA   (live pipeline: posts -> features -> Newton LR + NN + tuned ensemble -> accuracy vs baselines)",
                        "python src/demo.py --stock GME --title \"...\" --score 1200 --comments 340",
                        "python src/demo.py --stock TSLA --task volatility --title \"...\"",
                        "python src/demo.py --replay GME",
                        "(run_experiments.py produces all of results/ and models/; several minutes)"])
    prs.core_properties.author = prs.core_properties.last_modified_by = "Prithvi Aithal, Sagar A Walikar"
    prs.core_properties.title = "WSB market prediction - mini-project"
    prs.save(path)


if __name__ == "__main__":
    build_pdf(os.path.join(HERE, "writeup.pdf"))
    build_pptx(os.path.join(HERE, "presentation.pptx"))
    print("built docs/writeup.pdf and docs/presentation.pptx")
