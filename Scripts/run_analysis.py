"""
DS 4002 — Sentiment Analysis Case Study
Full analysis: VADER vs. HuggingFace RoBERTa on Letterboxd reviews
Run from repo root: python SCRIPTS/run_analysis.py
"""

# ── Step 1: Imports ───────────────────────────────────────────────────────────
import os
import sys

import nltk
nltk.download("vader_lexicon", quiet=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

os.makedirs("figures", exist_ok=True)
print("Step 1 complete: dependencies loaded.\n")

# ── Step 2: Load data ─────────────────────────────────────────────────────────
df = pd.read_csv("DATA/letterboxd_reviews.csv")
print(f"Step 2: Loaded {df.shape[0]:,} reviews × {df.shape[1]} columns.")
print("True-label distribution:")
print(df["true_label"].value_counts().to_string())
print()

# ── Step 3: Run VADER ─────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

def vader_classify(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    score = sia.polarity_scores(text)["compound"]
    if score < -0.05:
        return "negative"
    elif score > 0.05:
        return "positive"
    return "neutral"

df["vader_compound"] = df["review_text"].apply(
    lambda t: sia.polarity_scores(t)["compound"] if isinstance(t, str) else 0.0
)
df["vader_label"] = df["vader_compound"].apply(
    lambda c: "negative" if c < -0.05 else ("positive" if c > 0.05 else "neutral")
)

df.to_csv("DATA/letterboxd_reviews_with_vader.csv", index=False)
print("Step 3: VADER labels computed and saved to DATA/letterboxd_reviews_with_vader.csv.")
print("VADER prediction distribution:")
print(df["vader_label"].value_counts().to_string())
print()

# ── Step 4: Evaluate both models ──────────────────────────────────────────────
LABELS = ["negative", "neutral", "positive"]
y_true = df["true_label"]
y_vader = df["vader_label"]
y_hf = df["pred_label_thresh"]

vader_acc = accuracy_score(y_true, y_vader)
vader_f1  = f1_score(y_true, y_vader, average="macro", labels=LABELS, zero_division=0)
hf_acc    = accuracy_score(y_true, y_hf)
hf_f1     = f1_score(y_true, y_hf, average="macro", labels=LABELS, zero_division=0)

print("=" * 60)
print("VADER Results")
print("=" * 60)
print(f"  Accuracy  : {vader_acc:.4f} ({vader_acc*100:.1f}%)")
print(f"  Macro-F1  : {vader_f1:.4f} ({vader_f1*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_true, y_vader, labels=LABELS, zero_division=0))

print("=" * 60)
print("HuggingFace RoBERTa Results")
print("=" * 60)
print(f"  Accuracy  : {hf_acc:.4f} ({hf_acc*100:.1f}%)")
print(f"  Macro-F1  : {hf_f1:.4f} ({hf_f1*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_true, y_hf, labels=LABELS, zero_division=0))

cm_vader = confusion_matrix(y_true, y_vader, labels=LABELS)
cm_hf    = confusion_matrix(y_true, y_hf, labels=LABELS)

print("VADER Confusion Matrix (rows=true, cols=pred, order=neg/neu/pos):")
print(cm_vader)
print("\nHuggingFace Confusion Matrix:")
print(cm_hf)
print()

# ── Step 5: Figures ───────────────────────────────────────────────────────────

# Shared style
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Figure 1: side-by-side bar chart of accuracy & macro-F1 ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))

metrics      = ["Accuracy", "Macro-F1"]
vader_vals   = [vader_acc * 100, vader_f1 * 100]
hf_vals      = [hf_acc * 100,    hf_f1 * 100]

x     = np.arange(len(metrics))
width = 0.32

bars1 = ax.bar(x - width / 2, vader_vals, width, label="VADER",         color="#4C72B0")
bars2 = ax.bar(x + width / 2, hf_vals,    width, label="HuggingFace",   color="#DD8452")

for bar, val in zip(list(bars1) + list(bars2), vader_vals + hf_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_ylim(0, max(vader_vals + hf_vals) + 12)
ax.set_title(
    "VADER vs. HuggingFace RoBERTa: Performance on Letterboxd Reviews",
    fontsize=12, fontweight="bold", pad=14,
)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig("figures/fig1_model_comparison.png", dpi=150)
plt.close(fig)
print("Saved figures/fig1_model_comparison.png")

# ── Figure 2: confusion matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, name, acc in [
    (axes[0], cm_vader, "VADER",         vader_acc),
    (axes[1], cm_hf,    "HuggingFace",   hf_acc),
]:
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{name}  (Accuracy {acc*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xticklabels([l.capitalize() for l in LABELS], rotation=30, ha="right")
    ax.set_yticklabels([l.capitalize() for l in LABELS], rotation=0)

fig.suptitle("Confusion Matrices — Negative / Neutral / Positive", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("figures/fig2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig2_confusion_matrices.png")

# ── Figure 3: accuracy by review word-count quartile ─────────────────────────
df["wc_quartile"] = pd.qcut(
    df["review_word_count"],
    q=4,
    labels=["Q1 (short)", "Q2", "Q3", "Q4 (long)"],
    duplicates="drop",
)

wc_vader = df.groupby("wc_quartile", observed=True).apply(
    lambda g: accuracy_score(g["true_label"], g["vader_label"]) * 100,
    include_groups=False,
)
wc_hf = df.groupby("wc_quartile", observed=True).apply(
    lambda g: accuracy_score(g["true_label"], g["pred_label_thresh"]) * 100,
    include_groups=False,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wc_vader.index.astype(str), wc_vader.values, marker="o", label="VADER",       color="#4C72B0", linewidth=2)
ax.plot(wc_hf.index.astype(str),    wc_hf.values,    marker="s", label="HuggingFace", color="#DD8452", linewidth=2)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.set_xlabel("Review Length Quartile (by word count)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Model Accuracy by Review Length", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig("figures/fig3_accuracy_by_length.png", dpi=150)
plt.close(fig)
print("Saved figures/fig3_accuracy_by_length.png")

# ── Figure 4: accuracy by emoji presence ─────────────────────────────────────
emoji_groups = {True: "Has Emoji", False: "No Emoji"}
emoji_vader, emoji_hf = [], []

for flag in [True, False]:
    sub = df[df["has_emoji"] == flag]
    emoji_vader.append(accuracy_score(sub["true_label"], sub["vader_label"]) * 100)
    emoji_hf.append(accuracy_score(sub["true_label"], sub["pred_label_thresh"]) * 100)

fig, ax = plt.subplots(figsize=(7, 5))
x     = np.arange(2)
width = 0.32
labels_x = ["Has Emoji", "No Emoji"]

bars1 = ax.bar(x - width / 2, emoji_vader, width, label="VADER",       color="#4C72B0")
bars2 = ax.bar(x + width / 2, emoji_hf,   width, label="HuggingFace", color="#DD8452")

for bar, val in zip(list(bars1) + list(bars2), emoji_vader + emoji_hf):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels_x, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, max(emoji_vader + emoji_hf) + 12)
ax.set_title("Model Accuracy by Emoji Presence", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig("figures/fig4_accuracy_by_emoji.png", dpi=150)
plt.close(fig)
print("Saved figures/fig4_accuracy_by_emoji.png")

# ── Figure 5: VADER compound score KDE by true label ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

palette = {"negative": "#E05C5C", "neutral": "#F0A500", "positive": "#4CAF50"}

for label in ["negative", "neutral", "positive"]:
    subset = df[df["true_label"] == label]["vader_compound"]
    subset.plot.kde(ax=ax, label=label.capitalize(), color=palette[label], linewidth=2.5)

ax.axvline(-0.05, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Threshold (±0.05)")
ax.axvline( 0.05, color="grey", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlim(-1, 1)
ax.set_xlabel("VADER Compound Score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("VADER Score Distribution by True Sentiment Label", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig("figures/fig5_vader_score_distribution.png", dpi=150)
plt.close(fig)
print("Saved figures/fig5_vader_score_distribution.png")

# ── Figure 6: prediction distribution comparison ─────────────────────────────
n = len(df)
sources = {
    "True Labels":   df["true_label"].value_counts(normalize=True) * 100,
    "VADER":         df["vader_label"].value_counts(normalize=True) * 100,
    "HuggingFace":   df["pred_label_thresh"].value_counts(normalize=True) * 100,
}

source_names = list(sources.keys())
class_order  = ["negative", "neutral", "positive"]
colors       = {"negative": "#E05C5C", "neutral": "#F0A500", "positive": "#4CAF50"}

x     = np.arange(len(source_names))
width = 0.24

fig, ax = plt.subplots(figsize=(9, 5))

for i, cls in enumerate(class_order):
    vals = [sources[src].get(cls, 0) for src in source_names]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=cls.capitalize(), color=colors[cls])
    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

ax.set_xticks(x)
ax.set_xticklabels(source_names, fontsize=12)
ax.set_ylabel("Proportion of Predictions (%)", fontsize=11)
ax.set_title("Prediction Distribution: True Labels vs. Model Outputs", fontsize=12, fontweight="bold")
ax.legend(title="Class", fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig6_prediction_distributions.png", dpi=150)
plt.close(fig)
print("Saved figures/fig6_prediction_distributions.png")

print("\nAll figures saved. Analysis complete.")
print(f"\nSummary:")
print(f"  VADER       — Accuracy: {vader_acc*100:.1f}%  |  Macro-F1: {vader_f1*100:.1f}%")
print(f"  HuggingFace — Accuracy: {hf_acc*100:.1f}%  |  Macro-F1: {hf_f1*100:.1f}%")
