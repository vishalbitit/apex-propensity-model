"""
generate_dataset.py
────────────────────────────────────────────────────────────────────────────
Generates a realistic synthetic dataset of Apex Securities customers for
training a Mutual Fund Purchase Propensity model.

Features mirror real-world CRM / trading system data:
  • Demographics    : age, gender, city_tier, income_bracket
  • Account info    : account_type, account_age_months, kyc_status
  • Trading behaviour: avg_monthly_trades, last_trade_days_ago, primary_segment
  • Portfolio       : portfolio_value, avg_trade_value, total_investment
  • Engagement      : login_frequency, app_usage_days, research reports, webinars
  • Product history : sip_count, equity_holdings_count, ipo_applications
  • Service         : customer_service_calls, margin_utilisation_pct, referrals
  • Risk profile    : risk_appetite, relationship_manager flag

Target: will_buy_mutual_fund  (1 = purchased within 90 days, 0 = did not)

Usage:
    python generate_dataset.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
from config import (
    DATA_DIR, RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    N_CUSTOMERS, RANDOM_SEED, TEST_SIZE, TARGET_COLUMN
)
from sklearn.model_selection import train_test_split


def generate_customers(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = rng.integers(22, 70, n)
    gender = rng.choice(["Male", "Female"], n, p=[0.68, 0.32])
    city_tier = rng.choice(["Tier1", "Tier2", "Tier3"], n, p=[0.45, 0.35, 0.20])
    income_bracket = rng.choice(
        ["<5L", "5-10L", "10-25L", "25-50L", ">50L"], n,
        p=[0.15, 0.30, 0.30, 0.15, 0.10]
    )

    # ── Account Info ──────────────────────────────────────────────────────────
    account_type = rng.choice(
        ["Demat+Trading", "Demat Only", "Trading Only"], n,
        p=[0.55, 0.25, 0.20]
    )
    account_age_months = rng.integers(1, 180, n)
    kyc_status = rng.choice(["Complete", "Pending", "Expired"], n, p=[0.85, 0.10, 0.05])

    # ── Risk Profile ──────────────────────────────────────────────────────────
    risk_appetite = rng.choice(
        ["Conservative", "Moderate", "Aggressive"], n, p=[0.30, 0.45, 0.25]
    )
    relationship_manager = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])

    # ── Trading Behaviour ─────────────────────────────────────────────────────
    avg_monthly_trades = np.clip(rng.exponential(8, n), 0, 200).astype(int)
    last_trade_days_ago = rng.integers(0, 365, n)
    primary_segment = rng.choice(
        ["Equity", "Derivatives", "Mutual Funds", "IPO", "Bonds"], n,
        p=[0.45, 0.20, 0.18, 0.12, 0.05]
    )

    # ── Portfolio / Financial ─────────────────────────────────────────────────
    portfolio_value = np.clip(rng.lognormal(11, 2, n), 1_000, 5_00_00_000).round(2)
    avg_trade_value = np.clip(rng.lognormal(9, 1.5, n), 500, 10_00_000).round(2)
    total_investment = (portfolio_value * rng.uniform(0.5, 1.5, n)).round(2)

    # ── Engagement ────────────────────────────────────────────────────────────
    login_frequency_monthly = np.clip(rng.integers(0, 90, n), 0, 90)
    app_usage_days_monthly = np.clip(rng.integers(0, 30, n), 0, 30)
    research_reports_viewed = np.clip(rng.integers(0, 50, n), 0, 50)
    webinars_attended = np.clip(rng.integers(0, 20, n), 0, 20)

    # ── Product History ───────────────────────────────────────────────────────
    sip_count = rng.integers(0, 15, n)
    equity_holdings_count = rng.integers(0, 80, n)
    ipo_applications = rng.integers(0, 25, n)

    # ── Service & Others ──────────────────────────────────────────────────────
    customer_service_calls = rng.integers(0, 20, n)
    margin_utilisation_pct = np.clip(rng.uniform(0, 100, n), 0, 100).round(2)
    referral_count = rng.integers(0, 10, n)

    # ── Target: will_buy_mutual_fund ─────────────────────────────────────────
    # Propensity score driven by real business logic
    score = (
        0.10 * (sip_count / 15)
        + 0.08 * (webinars_attended / 20)
        + 0.08 * (research_reports_viewed / 50)
        + 0.07 * (login_frequency_monthly / 90)
        + 0.07 * (app_usage_days_monthly / 30)
        + 0.06 * (portfolio_value / portfolio_value.max())
        + 0.06 * ((account_age_months > 12).astype(float))
        + 0.05 * (ipo_applications / 25)
        + 0.05 * ((risk_appetite == "Conservative").astype(float))
        + 0.05 * ((primary_segment == "Mutual Funds").astype(float))
        + 0.04 * (np.isin(income_bracket, ["10-25L", "25-50L", ">50L"]).astype(float))
        + 0.04 * ((relationship_manager == "Yes").astype(float))
        + 0.03 * ((kyc_status == "Complete").astype(float))
        + 0.03 * (referral_count / 10)
        - 0.05 * (last_trade_days_ago / 365)          # dormant = less likely
        - 0.03 * (customer_service_calls / 20)         # complaints = less likely
    )

    # Add noise and convert to binary with ~28% positive rate
    noise = rng.normal(0, 0.08, n)
    prob = np.clip(score + noise, 0, 1)
    threshold = np.percentile(prob, 72)
    target = (prob >= threshold).astype(int)

    df = pd.DataFrame({
        # Demographics
        "age": age,
        "gender": gender,
        "city_tier": city_tier,
        "income_bracket": income_bracket,
        # Account
        "account_type": account_type,
        "account_age_months": account_age_months,
        "kyc_status": kyc_status,
        # Risk
        "risk_appetite": risk_appetite,
        "relationship_manager": relationship_manager,
        # Trading
        "avg_monthly_trades": avg_monthly_trades,
        "last_trade_days_ago": last_trade_days_ago,
        "primary_segment": primary_segment,
        # Portfolio
        "portfolio_value": portfolio_value,
        "avg_trade_value": avg_trade_value,
        "total_investment": total_investment,
        # Engagement
        "login_frequency_monthly": login_frequency_monthly,
        "app_usage_days_monthly": app_usage_days_monthly,
        "research_reports_viewed": research_reports_viewed,
        "webinars_attended": webinars_attended,
        # Products
        "sip_count": sip_count,
        "equity_holdings_count": equity_holdings_count,
        "ipo_applications": ipo_applications,
        # Service
        "customer_service_calls": customer_service_calls,
        "margin_utilisation_pct": margin_utilisation_pct,
        "referral_count": referral_count,
        # Target
        TARGET_COLUMN: target,
    })

    return df


def save_splits(df: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[✓] Raw dataset saved  → {RAW_DATA_PATH}  ({len(df):,} rows)")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN]
    )
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"[✓] Train split saved  → {TRAIN_DATA_PATH}  ({len(train_df):,} rows)")
    print(f"[✓] Test split saved   → {TEST_DATA_PATH}   ({len(test_df):,} rows)")

    pos_rate = df[TARGET_COLUMN].mean() * 100
    print(f"\n[i] Target distribution : {pos_rate:.1f}% positive  /  {100-pos_rate:.1f}% negative")
    print(f"[i] Feature columns     : {len(df.columns) - 1}")


def main():
    print("=" * 60)
    print("  Apex Securities — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"Generating {N_CUSTOMERS:,} customer records …")
    df = generate_customers()
    save_splits(df)
    print("\nDone! Run 'python train_model.py' next.")


if __name__ == "__main__":
    main()
