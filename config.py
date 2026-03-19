# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for Apex Securities Customer Propensity ML Model
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")

RAW_DATA_PATH   = os.path.join(DATA_DIR, "apex_customers.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH  = os.path.join(DATA_DIR, "test.csv")

XGB_MODEL_PATH  = os.path.join(MODEL_DIR, "xgboost_propensity.pkl")
LGB_MODEL_PATH  = os.path.join(MODEL_DIR, "lightgbm_propensity.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Dataset ───────────────────────────────────────────────────────────────────
N_CUSTOMERS     = 50_000      # number of synthetic customers to generate
RANDOM_SEED     = 42
TARGET_COLUMN   = "will_buy_mutual_fund"   # 1 = likely buyer, 0 = not likely

# ── Model Training ────────────────────────────────────────────────────────────
TEST_SIZE       = 0.20        # 80/20 train-test split
CV_FOLDS        = 5           # cross-validation folds

XGBOOST_PARAMS  = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "num_leaves":       63,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ── Feature Groups ────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender",
    "city_tier",
    "account_type",
    "risk_appetite",
    "income_bracket",
    "primary_segment",
    "kyc_status",
    "relationship_manager",
]

NUMERICAL_FEATURES = [
    "age",
    "account_age_months",
    "avg_monthly_trades",
    "last_trade_days_ago",
    "portfolio_value",
    "avg_trade_value",
    "total_investment",
    "login_frequency_monthly",
    "app_usage_days_monthly",
    "research_reports_viewed",
    "webinars_attended",
    "sip_count",
    "equity_holdings_count",
    "ipo_applications",
    "customer_service_calls",
    "margin_utilisation_pct",
    "referral_count",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
