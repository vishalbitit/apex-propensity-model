"""
evaluate_model.py
────────────────────────────────────────────────────────────────────────────
Evaluates both trained propensity models on the hold-out test set.
Generates:
  • Classification report (precision, recall, F1)
  • AUC-ROC & AUC-PR scores
  • Confusion matrix
  • ROC curve plot
  • Precision-Recall curve plot
  • Feature importance plot (top-20)
  • SHAP summary plot (XGBoost)
  • Propensity score decile analysis (lift / gain table)
  • All plots saved to reports/

Usage:
    python evaluate_model.py [--model xgb|lgb|both]
────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder

import config

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Load Artifacts ────────────────────────────────────────────────────────────

def load_artifacts():
    required = [config.XGB_MODEL_PATH, config.LGB_MODEL_PATH,
                config.SCALER_PATH, config.TEST_DATA_PATH]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}\nRun generate_dataset.py then train_model.py first.")

    xgb      = joblib.load(config.XGB_MODEL_PATH)
    lgb      = joblib.load(config.LGB_MODEL_PATH)
    scaler   = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(os.path.join(config.MODEL_DIR, "encoders.pkl"))
    test_df  = pd.read_csv(config.TEST_DATA_PATH)
    print("[✓] Artifacts loaded")
    return xgb, lgb, scaler, encoders, test_df


def preprocess_test(test_df: pd.DataFrame, encoders, scaler):
    df = test_df.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))
    X = df[config.ALL_FEATURES].values
    y = df[config.TARGET_COLUMN].values
    num_idx = [config.ALL_FEATURES.index(f) for f in config.NUMERICAL_FEATURES]
    X[:, num_idx] = scaler.transform(X[:, num_idx])
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(name: str, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name} — Test Set Metrics")
    print(f"{'─'*50}")
    print(f"  AUC-ROC           : {auc_roc:.4f}")
    print(f"  AUC-PR            : {auc_pr:.4f}")
    print(f"  F1-Score (thr=0.5): {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Buyer','Buyer'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n  {cm}")
    return y_prob


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc(models_probs: dict, y_test, outdir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Apex Propensity Models")
    ax.legend(loc="lower right")
    path = os.path.join(outdir, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {path}")


def plot_pr(models_probs: dict, y_test, outdir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in models_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, label=f"{name}  (AP={ap:.3f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve — Apex Propensity Models")
    ax.legend(loc="upper right")
    path = os.path.join(outdir, "pr_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {path}")


def plot_feature_importance(model, model_name: str, outdir: str):
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=config.ALL_FEATURES)
        fi_top = fi.nlargest(20).sort_values()

        fig, ax = plt.subplots(figsize=(8, 7))
        fi_top.plot(kind="barh", ax=ax, color="steelblue")
        ax.set(title=f"Top-20 Feature Importances — {model_name}",
               xlabel="Importance Score")
        path = os.path.join(outdir, f"feature_importance_{model_name.lower().replace(' ','_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] Saved: {path}")


def plot_shap(model, X_test, outdir: str):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:500])   # sample 500 rows for speed

        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(
            shap_values, X_test[:500],
            feature_names=config.ALL_FEATURES,
            show=False, plot_size=None
        )
        plt.title("SHAP Summary — XGBoost Propensity Model")
        path = os.path.join(outdir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved: {path}")
    except Exception as e:
        print(f"[!] SHAP plot skipped: {e}")


def decile_analysis(y_test, y_prob, model_name: str, outdir: str):
    df = pd.DataFrame({"actual": y_test, "score": y_prob})
    df["decile"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop")
    df["decile"] = 10 - df["decile"]   # decile 1 = highest score

    table = (
        df.groupby("decile")
          .agg(customers=("actual", "count"),
               buyers=("actual", "sum"),
               avg_score=("score", "mean"))
          .assign(
              conversion_rate=lambda x: x["buyers"] / x["customers"] * 100,
              cumulative_buyers=lambda x: x["buyers"].cumsum(),
          )
    )
    table["cumulative_lift"] = (
        table["cumulative_buyers"] / table["cumulative_buyers"].iloc[-1]
        / (table["customers"].cumsum() / table["customers"].sum())
    )

    path = os.path.join(outdir, f"decile_analysis_{model_name.lower().replace(' ','_')}.csv")
    table.to_csv(path)
    print(f"[✓] Saved: {path}")
    print(f"\n  Decile Analysis — {model_name} (top decile targets ~{table['conversion_rate'].iloc[0]:.1f}% buyers)")
    print(table[["customers", "buyers", "avg_score", "conversion_rate", "cumulative_lift"]].to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main(which: str = "both"):
    print("=" * 60)
    print("  Apex Securities — Model Evaluation")
    print("=" * 60)

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    xgb, lgb, scaler, encoders, test_df = load_artifacts()
    X_test, y_test = preprocess_test(test_df, encoders, scaler)

    models_probs = {}

    if which in ("xgb", "both"):
        prob_xgb = print_metrics("XGBoost", xgb, X_test, y_test)
        models_probs["XGBoost"] = prob_xgb
        plot_feature_importance(xgb, "XGBoost", config.REPORTS_DIR)
        plot_shap(xgb, X_test, config.REPORTS_DIR)
        decile_analysis(y_test, prob_xgb, "XGBoost", config.REPORTS_DIR)

    if which in ("lgb", "both"):
        prob_lgb = print_metrics("LightGBM", lgb, X_test, y_test)
        models_probs["LightGBM"] = prob_lgb
        plot_feature_importance(lgb, "LightGBM", config.REPORTS_DIR)
        decile_analysis(y_test, prob_lgb, "LightGBM", config.REPORTS_DIR)

    # Comparison plots
    plot_roc(models_probs, y_test, config.REPORTS_DIR)
    plot_pr(models_probs, y_test, config.REPORTS_DIR)

    print(f"\n[✓] All reports saved → {config.REPORTS_DIR}/")
    print("\nDone! Run 'python predict.py' to score new customers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "lgb", "both"], default="both")
    args = parser.parse_args()
    main(which=args.model)
