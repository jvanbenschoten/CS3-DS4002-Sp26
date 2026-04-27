"""
DS 4002 — Sentiment Analysis Case Study
Starter Script for Students

This script walks through comparing VADER and HuggingFace RoBERTa on
Letterboxd movie reviews. Steps 1-3 are fully implemented for you.
Steps 4-6 are left as TODOs with hints to guide your work.

Run from the repo root:
    python SCRIPTS/starter_script.py
"""

# ── Step 1: Load the data ─────────────────────────────────────────────────────
# We use pandas to load the CSV into a DataFrame — think of it like a spreadsheet
# that Python can work with programmatically.

import pandas as pd

df = pd.read_csv("DATA/letterboxd_reviews.csv")

print(f"Loaded {len(df):,} reviews.")
print(f"Columns available: {list(df.columns)}\n")

# Let's look at the first few rows to understand the structure
print("Sample rows:")
print(df[["review_text", "star_rating", "true_label", "pred_label_thresh"]].head(3))
print()

# ── Step 2: Understand the ground truth labels ────────────────────────────────
# The column `true_label` is our "correct answer" — derived from the star rating:
#   - star_rating <= 1.0  → "negative"
#   - star_rating 1.5–3.0 → "neutral"
#   - star_rating >= 3.5  → "positive"
#
# This is what we'll compare both models against.

print("True label distribution:")
print(df["true_label"].value_counts())
print()

# Notice anything? The dataset is heavily imbalanced — ~93% of reviews are
# "positive". This matters a lot for choosing the right evaluation metric.
# A model that predicts "positive" for everything would get 93% accuracy —
# but that's not useful. This is why we use Macro-F1 as our primary metric.

# ── Step 3: Run VADER ─────────────────────────────────────────────────────────
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
# sentiment tool built for social media text. It returns a "compound" score
# between -1 (most negative) and +1 (most positive).
#
# Classification thresholds (from the original VADER paper):
#   compound < -0.05  → "negative"
#   compound >  0.05  → "positive"
#   otherwise         → "neutral"

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Compute compound score for each review
df["vader_compound"] = df["review_text"].apply(
    lambda text: sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0.0
)

# Convert compound score to a label
def classify_vader(compound_score):
    if compound_score < -0.05:
        return "negative"
    elif compound_score > 0.05:
        return "positive"
    else:
        return "neutral"

df["vader_label"] = df["vader_compound"].apply(classify_vader)

print("VADER prediction distribution:")
print(df["vader_label"].value_counts())
print()

# Let's look at a few examples to get a feel for how VADER works
print("Sample VADER predictions:")
sample = df[["review_text", "star_rating", "true_label", "vader_compound", "vader_label"]].sample(5, random_state=42)
for _, row in sample.iterrows():
    snippet = str(row["review_text"])[:80] + "..."
    print(f"  [{row['true_label']:8s} | VADER: {row['vader_label']:8s} | score: {row['vader_compound']:+.3f}]  {snippet}")
print()

# ── Step 4: TODO — Evaluate VADER ────────────────────────────────────────────
# Your task: compute accuracy and Macro-F1 for VADER, then print results.
#
# Hints:
#   - Import from sklearn.metrics: accuracy_score, f1_score, classification_report
#   - y_true = df["true_label"]
#   - y_vader = df["vader_label"]
#   - For f1_score, set average="macro" and zero_division=0
#   - Use classification_report to see per-class precision, recall, F1
#
# Example skeleton:
#
#   from sklearn.metrics import accuracy_score, f1_score, classification_report
#
#   vader_acc = accuracy_score(y_true, y_vader)
#   vader_f1  = f1_score(y_true, y_vader, average="macro", zero_division=0)
#   print(f"VADER Accuracy: {vader_acc:.4f}")
#   print(f"VADER Macro-F1: {vader_f1:.4f}")
#   print(classification_report(y_true, y_vader, zero_division=0))

# TODO: your code here


# ── Step 5: TODO — Compare HuggingFace results ────────────────────────────────
# The HuggingFace predictions are already in the column `pred_label_thresh`.
# Do NOT re-run the model — use the column directly.
#
# Your task: compute the same accuracy and Macro-F1 for HuggingFace, then
# print a side-by-side comparison of both models.
#
# Hints:
#   - y_hf = df["pred_label_thresh"]
#   - Reuse accuracy_score, f1_score, classification_report from Step 4
#   - Build a small summary table with pandas:
#
#     summary = pd.DataFrame({
#         "Model":    ["VADER", "HuggingFace"],
#         "Accuracy": [vader_acc, hf_acc],
#         "Macro-F1": [vader_f1,  hf_f1],
#     })
#     print(summary)
#
#   - Which metric should you focus on given the class imbalance? Why?

# TODO: your code here


# ── Step 6: TODO — Generate figures ──────────────────────────────────────────
# Your task: create at least two plots to visualize the comparison.
# Save them to the figures/ folder (create it first with os.makedirs).
#
# Suggested figures (you can add more!):
#
# Figure A — Bar chart comparing Accuracy and Macro-F1 for both models
#   Hint: use matplotlib.pyplot; look at plt.bar() with grouped bars
#
# Figure B — Confusion matrix for VADER (and optionally HuggingFace)
#   Hint: use seaborn.heatmap with the confusion_matrix from sklearn
#   Labels order: ["negative", "neutral", "positive"]
#
# Figure C (stretch) — KDE plot of vader_compound scores colored by true_label
#   Hint: df[df["true_label"] == "positive"]["vader_compound"].plot.kde(...)
#
# Skeleton for saving a figure:
#
#   import matplotlib
#   matplotlib.use("Agg")        # headless backend — needed for scripts
#   import matplotlib.pyplot as plt
#   import os
#
#   os.makedirs("figures", exist_ok=True)
#   fig, ax = plt.subplots(figsize=(8, 5))
#   # ... draw your plot ...
#   fig.savefig("figures/my_figure.png", dpi=150)
#   plt.close(fig)
#   print("Saved figures/my_figure.png")

# TODO: your code here
