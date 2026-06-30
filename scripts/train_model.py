"""
scripts/train_model.py
======================
Trains three ML models on the placement dataset, compares them,
saves the best model, and generates evaluation artefacts.

Why this script exists:
    Separating training from the Streamlit app means you can retrain
    offline, inspect evaluation charts, and version the model file
    independently of the UI.  This is how real ML projects work.

How to run:
    python scripts/train_model.py

Outputs:
    models/trained_model.pkl        — Best model (Random Forest)
    evaluation/confusion_matrix.png — Visual confusion matrix
    evaluation/classification_report.txt
    evaluation/accuracy_comparison.png
    evaluation/feature_importance.png
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_CSV: Path = PROJECT_ROOT / "data" / "placement_sample.csv"
MODEL_PKL: Path = PROJECT_ROOT / "models" / "trained_model.pkl"
EVAL_DIR: Path = PROJECT_ROOT / "evaluation"

FEATURE_COLUMNS: list[str] = [
    "cgpa",
    "projects",
    "coding_platform_activity",
    "internship_score",
    "skill_match_percentage",
]

# Human-readable labels for charts
FEATURE_LABELS: dict[str, str] = {
    "cgpa": "CGPA",
    "projects": "Projects",
    "coding_platform_activity": "Coding Activity",
    "internship_score": "Internship",
    "skill_match_percentage": "Skill Match",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load features and target from the placement CSV.

    Returns:
        X — feature DataFrame
        y — placed column (0/1)
    """
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_COLUMNS]
    y = df["placed"]
    return X, y


# ── Model definitions ─────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Return a dictionary of untrained model instances.

    Why these three?
        Logistic Regression — linear, very explainable, good baseline
        Decision Tree       — rule-based, easy to visualise, prone to overfit
        Random Forest       — ensemble of trees, best accuracy + feature importance
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Compute four standard classification metrics.

    Why all four?
        Accuracy alone is misleading when classes are imbalanced.
        Precision/Recall/F1 give a fuller picture of where the model fails.
    """
    preds = model.predict(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, preds) * 100, 1),
        "Precision": round(precision_score(y_test, preds, zero_division=0) * 100, 1),
        "Recall":    round(recall_score(y_test, preds, zero_division=0) * 100, 1),
        "F1 Score":  round(f1_score(y_test, preds, zero_division=0) * 100, 1),
    }


# ── Evaluation artefacts ─────────────────────────────────────────────────────

def save_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Save a colour-coded confusion matrix for the best model.

    What this shows:
        True Positives / True Negatives / False Positives / False Negatives.
        A good model has large numbers on the diagonal and small numbers off it.
    """
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed", "Placed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Best Model (Random Forest)", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  ✔  confusion_matrix.png saved")


def save_classification_report(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Write the full sklearn classification report to a text file.

    Why useful:
        Per-class precision/recall/F1 gives the full picture — especially
        important when one class is harder to predict than the other.
    """
    preds = model.predict(X_test)
    report_text = classification_report(
        y_test, preds,
        target_names=["Not Placed", "Placed"],
    )
    path = EVAL_DIR / "classification_report.txt"
    path.write_text(
        "Placement Mentor Lite — Classification Report\n"
        "==============================================\n\n"
        + report_text
    )
    print("  ✔  classification_report.txt saved")


def save_accuracy_comparison(results: dict[str, dict]) -> None:
    """
    Bar chart comparing all three models on the four metrics.

    Why this chart:
        A visual table lets interviewers and faculty see at a glance which
        model is stronger and by how much.
    """
    models = list(results.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(models))
    width = 0.20
    colors = ["#287c8e", "#55a868", "#c44e52", "#8172b2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val}%",
                ha="center", va="bottom", fontsize=7.5
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy / Precision / Recall / F1", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✔  accuracy_comparison.png saved")


def save_feature_importance(rf_model: RandomForestClassifier) -> None:
    """
    Horizontal bar chart of Random Forest feature importances.

    Why Random Forest for this:
        RF natively computes how much each feature reduces impurity
        across all decision trees, giving a reliable importance ranking.
    """
    importances = rf_model.feature_importances_
    labels = [FEATURE_LABELS[col] for col in FEATURE_COLUMNS]
    pct = [round(float(imp * 100), 1) for imp in importances]

    # Sort ascending so highest bar appears at the top
    sorted_pairs = sorted(zip(labels, pct), key=lambda t: t[1])
    labels_sorted, pct_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels_sorted, pct_sorted, color="#287c8e")
    for bar, val in zip(bars, pct_sorted):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9)
    ax.set_xlabel("Importance (%)")
    ax.set_title("Feature Importance — Random Forest", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  ✔  feature_importance.png saved")


# ── Main training pipeline ────────────────────────────────────────────────────

def main() -> None:
    """
    Full training pipeline:
        1. Load data
        2. Split into train/test
        3. Train all three models
        4. Evaluate and compare
        5. Save best model
        6. Save all evaluation artefacts
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 52)
    print("  Placement Mentor Lite — Model Training")
    print("=" * 52)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    X, y = load_data()
    print(f"\n  Dataset        : {len(X)} records, {y.sum()} placed")

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    # stratify=y keeps the placed/not-placed ratio equal in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"  Training set   : {len(X_train)} rows")
    print(f"  Test set       : {len(X_test)} rows")

    # ── Step 3: Train all models ──────────────────────────────────────────────
    models = build_models()
    trained: dict = {}
    results: dict[str, dict] = {}

    print("\n  Training models…")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        trained[name] = clf
        results[name] = evaluate_model(clf, X_test, y_test)
        print(f"  ✔  {name}")

    # ── Step 4: Print comparison table ───────────────────────────────────────
    print("\n  Model Comparison:")
    print(f"  {'Model':<24} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 62)
    for model_name, metrics in results.items():
        print(
            f"  {model_name:<24}"
            f" {metrics['Accuracy']:>8}%"
            f" {metrics['Precision']:>9}%"
            f" {metrics['Recall']:>7}%"
            f" {metrics['F1 Score']:>7}%"
        )

    # ── Step 5: Select best model by F1 (more robust than accuracy alone) ────
    best_name: str = max(results, key=lambda m: results[m]["F1 Score"])
    best_model = trained[best_name]
    print(f"\n  Best model     : {best_name} (F1 = {results[best_name]['F1 Score']}%)")

    # ── Step 6: Save model ────────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Model saved    : {MODEL_PKL}")

    # ── Step 7: Save evaluation artefacts ────────────────────────────────────
    print("\n  Saving evaluation artefacts…")
    save_confusion_matrix(best_model, X_test, y_test)
    save_classification_report(best_model, X_test, y_test)
    save_accuracy_comparison(results)
    save_feature_importance(trained["Random Forest"])

    print(f"\n  All artefacts  : {EVAL_DIR}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
