"""
predict.py
────────────────────────────────────────────────────────────────────────────
Scores new / unseen customers using the trained XGBoost & LightGBM models.
Produces:
  • propensity_score    : ensemble probability (avg of XGB + LGB)
  • xgb_score / lgb_score : individual model scores
  • propensity_segment  : HIGH / MEDIUM / LOW based on threshold
  • rank                : customers ranked 1 = most likely to buy

Can accept:
  (a) A CSV file path (--input path/to/new_customers.csv)
  (b) No argument      → scores a fresh synthetic sample of 200 customers

Usage:
    python predict.py
    python predict.py --input data/my_customers.csv --model xgb
    python predict.py --input data/my_customers.csv --output results/scored.csv
────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

import config
from generate_dataset import generate_customers

warnings.filterwarnings("ignore")


# ── Thresholds for segmentation ───────────────────────────────────────────────
HIGH_THRESHOLD   = 0.65    # P(buy) >= 0.65  → HIGH propensity
MEDIUM_THRESHOLD = 0.35    # P(buy) >= 0.35  → MEDIUM propensity
                           # else            → LOW propensity


# ── Load Artifacts ────────────────────────────────────────────────────────────

def load_artifacts():
    for p in [config.XGB_MODEL_PATH, config.LGB_MODEL_PATH,
              config.SCALER_PATH,
              os.path.join(config.MODEL_DIR, "encoders.pkl")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model artifact not found: {p}\n"
                "Please run:\n  python generate_dataset.py\n  python train_model.py"
            )
    xgb      = joblib.load(config.XGB_MODEL_PATH)
    lgb      = joblib.load(config.LGB_MODEL_PATH)
    scaler   = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(os.path.join(config.MODEL_DIR, "encoders.pkl"))
    print("[✓] Models loaded")
    return xgb, lgb, scaler, encoders


def preprocess_input(df: pd.DataFrame, encoders, scaler) -> np.ndarray:
    """Encode + scale features exactly as done during training."""
    df = df.copy()

    # Handle unseen categories gracefully
    for col, le in encoders.items():
        if col in df.columns:
            # Map unknown values to the most frequent class
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Fill missing numerical features with 0
    for col in config.NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[config.ALL_FEATURES].values.astype(float)
    num_idx = [config.ALL_FEATURES.index(f) for f in config.NUMERICAL_FEATURES]
    X[:, num_idx] = scaler.transform(X[:, num_idx])
    return X


def segment(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_customers(
    df_raw: pd.DataFrame,
    xgb, lgb, scaler, encoders,
    model: str = "both"
) -> pd.DataFrame:
    X = preprocess_input(df_raw, encoders, scaler)

    results = df_raw.copy()

    if model in ("xgb", "both"):
        xgb_prob = xgb.predict_proba(X)[:, 1]
        results["xgb_score"] = np.round(xgb_prob, 4)

    if model in ("lgb", "both"):
        lgb_prob = lgb.predict_proba(X)[:, 1]
        results["lgb_score"] = np.round(lgb_prob, 4)

    if model == "both":
        results["propensity_score"] = np.round((xgb_prob + lgb_prob) / 2, 4)
    elif model == "xgb":
        results["propensity_score"] = results["xgb_score"]
    else:
        results["propensity_score"] = results["lgb_score"]

    results["propensity_segment"] = results["propensity_score"].apply(segment)
    results["rank"] = results["propensity_score"].rank(ascending=False).astype(int)
    results = results.sort_values("rank")

    return results


def print_summary(scored_df: pd.DataFrame):
    seg_counts = scored_df["propensity_segment"].value_counts()
    total = len(scored_df)
    print(f"\n{'─'*50}")
    print(f"  Scoring Summary  ({total:,} customers)")
    print(f"{'─'*50}")
    for seg in ["HIGH", "MEDIUM", "LOW"]:
        n = seg_counts.get(seg, 0)
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {seg:<8} : {n:>5,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Avg propensity score : {scored_df['propensity_score'].mean():.3f}")
    print(f"\n  Top 10 customers by propensity score:")
    cols_to_show = ["rank", "propensity_score", "propensity_segment"]
    if "xgb_score" in scored_df.columns:
        cols_to_show += ["xgb_score"]
    if "lgb_score" in scored_df.columns:
        cols_to_show += ["lgb_score"]
    print(scored_df[cols_to_show].head(10).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score Apex Securities customers for MF purchase propensity"
    )
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to input CSV. Omit to use synthetic sample.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save scored output CSV.")
    parser.add_argument("--model",  choices=["xgb", "lgb", "both"], default="both",
                        help="Which model to use (default: both = ensemble)")
    parser.add_argument("--n",      type=int, default=200,
                        help="Number of synthetic customers if no --input given")
    args = parser.parse_args()

    print("=" * 60)
    print("  Apex Securities — Customer Propensity Scorer")
    print("=" * 60)

    xgb, lgb, scaler, encoders = load_artifacts()

    # ── Load input data ───────────────────────────────────────────────────────
    if args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        df_raw = pd.read_csv(args.input)
        # Drop target column if present
        if config.TARGET_COLUMN in df_raw.columns:
            df_raw = df_raw.drop(columns=[config.TARGET_COLUMN])
        print(f"[✓] Loaded {len(df_raw):,} customers from {args.input}")
    else:
        print(f"[i] No --input given. Generating {args.n} synthetic customers …")
        df_raw = generate_customers(n=args.n, seed=999)
        if config.TARGET_COLUMN in df_raw.columns:
            df_raw = df_raw.drop(columns=[config.TARGET_COLUMN])

    # ── Score ─────────────────────────────────────────────────────────────────
    scored = score_customers(df_raw, xgb, lgb, scaler, encoders, model=args.model)
    print_summary(scored)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output or os.path.join(config.DATA_DIR, "scored_customers.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"\n[✓] Scored output saved → {out_path}")


if __name__ == "__main__":
    main()
