"""
train_model.py
────────────────────────────────────────────────────────────────────────────
Trains XGBoost and LightGBM propensity models on the Apex Securities
customer dataset. Handles:
  • Preprocessing  : encoding categoricals, scaling numericals
  • Class imbalance: SMOTE oversampling
  • Model training : XGBoost + LightGBM with early stopping
  • Cross-validation: stratified k-fold AUC-ROC scoring
  • Persistence    : saves trained models + scaler to disk

Usage:
    python train_model.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection  import StratifiedKFold, cross_val_score
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import roc_auc_score
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier
from lightgbm                 import LGBMClassifier

import config

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {config.TRAIN_DATA_PATH}.\n"
            "Please run: python generate_dataset.py"
        )
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df  = pd.read_csv(config.TEST_DATA_PATH)
    print(f"[✓] Loaded train ({len(train_df):,} rows) | test ({len(test_df):,} rows)")
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Encode categoricals with LabelEncoder, scale numericals with StandardScaler.
    Returns (X_train, y_train, X_test, y_test, encoders, scaler).
    """
    encoders = {}

    # Label-encode categorical columns (fit on train, transform both)
    for col in config.CATEGORICAL_FEATURES:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
        encoders[col] = le

    X_train = train_df[config.ALL_FEATURES].values
    y_train = train_df[config.TARGET_COLUMN].values
    X_test  = test_df[config.ALL_FEATURES].values
    y_test  = test_df[config.TARGET_COLUMN].values

    # Scale numerical features
    scaler = StandardScaler()
    num_idx = [config.ALL_FEATURES.index(f) for f in config.NUMERICAL_FEATURES]
    X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_test[:, num_idx]  = scaler.transform(X_test[:, num_idx])

    print(f"[✓] Preprocessing done  | features={X_train.shape[1]}")
    print(f"    Train class balance : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"    Test  class balance : {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    return X_train, y_train, X_test, y_test, encoders, scaler


def apply_smote(X_train, y_train):
    """Oversample minority class to handle class imbalance."""
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[✓] SMOTE applied       | resampled train size={len(X_res):,}")
    return X_res, y_res


def cross_validate(model, X, y, name: str) -> float:
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"    [{name}] CV AUC-ROC: {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


# ── Model Builders ────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test) -> XGBClassifier:
    print("\n── XGBoost ──────────────────────────────────────")
    t0 = time.time()

    model = XGBClassifier(**config.XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"[✓] XGBoost test AUC-ROC : {auc:.4f}  ({time.time()-t0:.1f}s)")
    return model


def train_lightgbm(X_train, y_train, X_test, y_test) -> LGBMClassifier:
    print("\n── LightGBM ─────────────────────────────────────")
    t0 = time.time()

    model = LGBMClassifier(**config.LIGHTGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=None,
    )

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"[✓] LightGBM test AUC-ROC: {auc:.4f}  ({time.time()-t0:.1f}s)")
    return model


# ── Persistence ───────────────────────────────────────────────────────────────

def save_artifacts(xgb_model, lgb_model, scaler, encoders):
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    joblib.dump(xgb_model, config.XGB_MODEL_PATH)
    joblib.dump(lgb_model, config.LGB_MODEL_PATH)
    joblib.dump(scaler,    config.SCALER_PATH)
    joblib.dump(encoders,  os.path.join(config.MODEL_DIR, "encoders.pkl"))

    print(f"\n[✓] Models saved → {config.MODEL_DIR}/")
    print(f"    xgboost_propensity.pkl")
    print(f"    lightgbm_propensity.pkl")
    print(f"    scaler.pkl")
    print(f"    encoders.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Apex Securities — Propensity Model Training")
    print("=" * 60)

    train_df, test_df = load_data()

    print("\n[1/4] Preprocessing …")
    X_train, y_train, X_test, y_test, encoders, scaler = preprocess(train_df, test_df)

    print("\n[2/4] Applying SMOTE …")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\n[3/4] Cross-validating …")
    cross_validate(XGBClassifier(**config.XGBOOST_PARAMS),  X_train_res, y_train_res, "XGBoost")
    cross_validate(LGBMClassifier(**config.LIGHTGBM_PARAMS), X_train_res, y_train_res, "LightGBM")

    print("\n[4/4] Training final models on full training set …")
    xgb_model = train_xgboost(X_train_res, y_train_res, X_test, y_test)
    lgb_model  = train_lightgbm(X_train_res, y_train_res, X_test, y_test)

    save_artifacts(xgb_model, lgb_model, scaler, encoders)

    print("\nDone! Run 'python evaluate_model.py' to see detailed metrics and plots.")


if __name__ == "__main__":
    main()
