# Customer Propensity Model - Mutual Fund Purchase Prediction

A machine learning project that predicts which customers of a securities/broking firm are most likely to purchase a Mutual Fund product in the next 90 days.

Built using **XGBoost** and **LightGBM** on a synthetic dataset of 50,000 customers. The model outputs a propensity score (0–1) per customer, segments them into HIGH / MEDIUM / LOW tiers, and ranks them so sales teams can prioritise outreach efficiently.

---

## Why I Built This

Working closely with the financial services customers, I kept hearing about one of their problem: sales teams would run broad-based campaigns contacting thousands of customers, with very low conversion rates and a lot of wasted effort. The core issue was no systematic way to identify *which* customers were actually ready to buy at a given point in time.

This project is my attempt to solve that using ML - specifically propensity modelling. The idea is simple: instead of calling everyone, score every customer by likelihood to buy, and focus your effort on the top 20% who are most likely to convert. In practice this typically yields 2x+ the conversion rate for the same outreach spend.

The dataset here is synthetic (modelled on realistic patterns from the Indian retail broking industry), but the pipeline is designed to plug directly into a real CRM/trading system export with minimal changes.

---

## What the Model Does

Takes 25 customer features as input:

- **Who they are** - age, gender, city tier, income bracket
- **Account details** - account type, age of account, KYC status, risk appetite
- **Trading behaviour** - monthly trades, days since last trade, primary segment
- **Portfolio** - portfolio value, average trade size, total investment
- **Digital engagement** - login frequency, app usage, research reports read, webinars attended
- **Product history** - active SIPs, equity holdings, IPO applications
- **Service signals** - customer service calls, margin utilisation, referrals given

Outputs per customer:
- `propensity_score` - probability of purchase (0 to 1)
- `propensity_segment` - HIGH / MEDIUM / LOW
- `rank` - ranked from most to least likely buyer

---

## Project Structure

```
├── config.py             — paths, hyperparameters, feature lists
├── generate_dataset.py   — synthetic customer dataset generator
├── train_model.py        — preprocessing, SMOTE, training pipeline
├── evaluate_model.py     — metrics, plots, SHAP, decile analysis
├── predict.py            — score new customers
├── requirements.txt      — Python dependencies
├── Dockerfile            — containerise the project
├── docker-compose.yml    — easier Docker usage
├── docker-entrypoint.sh  — controls which step runs inside the container
│
├── data/                 (auto-created on first run)
├── models/               (auto-created on first run)
└── reports/              (auto-created on first run)
```

---

## Quickstart (Local Python)

```bash
# Clone the repo
git clone https://github.com/vishalbitit/apex-propensity-model.git
cd apex-propensity-model

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python generate_dataset.py
python train_model.py
python evaluate_model.py
python predict.py
```

---

## Quickstart (Docker)

No Python setup needed - just Docker installed.

```bash
# Clone the repo
git clone https://github.com/vishalbitit/apex-propensity-model.git
cd apex-propensity-model
# Build the image (first time only, or when code changes)
docker-compose build

# Run the full pipeline
docker-compose run propensity-model all

# Or run individual steps
docker-compose run propensity-model generate
docker-compose run propensity-model train
docker-compose run propensity-model evaluate
docker-compose run propensity-model predict
```

Generated files (`data/`, `models/`, `reports/`) are saved to your local folder automatically via Docker volumes.

---

## Step-by-Step Breakdown

### Step 1 - Generate Dataset
```bash
python generate_dataset.py
```
Generates 50,000 synthetic customers and saves train/test splits to `data/`.
The target column `will_buy_mutual_fund` is built using a weighted scoring formula
based on real business signals (SIP count, engagement, portfolio value, risk appetite etc.)

### Step 2 - Train Models
```bash
python train_model.py
```
- Label encodes categorical features, scales numericals with StandardScaler
- Applies SMOTE to handle class imbalance (28% buyers vs 72% non-buyers)
- Runs 5-fold stratified cross-validation
- Trains final XGBoost and LightGBM models
- Saves all artifacts to `models/`

### Step 3 - Evaluate
```bash
python evaluate_model.py
python evaluate_model.py --model xgb    # XGBoost only
python evaluate_model.py --model lgb    # LightGBM only
```
Outputs: AUC-ROC, AUC-PR, F1 score, confusion matrix, ROC/PR curve plots,
feature importance charts, SHAP summary plot, decile lift table

### Step 4 - Score New Customers
```bash
# Score a synthetic sample of 200 customers
python predict.py

# Score your own CSV
python predict.py --input data/my_customers.csv

# Use only XGBoost, save to a custom path
python predict.py --input data/my_customers.csv --model xgb --output results/scored.csv
```

---

## Input Format (for predict.py)

| Column | Type | Example |
|--------|------|---------|
| age | int | 35 |
| gender | str | Male / Female |
| city_tier | str | Tier1 / Tier2 / Tier3 |
| income_bracket | str | 5-10L / 10-25L / 25-50L / >50L / <5L |
| account_type | str | Demat+Trading / Demat Only / Trading Only |
| account_age_months | int | 36 |
| kyc_status | str | Complete / Pending / Expired |
| risk_appetite | str | Conservative / Moderate / Aggressive |
| relationship_manager | str | Yes / No |
| avg_monthly_trades | int | 12 |
| last_trade_days_ago | int | 30 |
| primary_segment | str | Equity / Derivatives / Mutual Funds / IPO / Bonds |
| portfolio_value | float | 250000.00 |
| avg_trade_value | float | 15000.00 |
| total_investment | float | 300000.00 |
| login_frequency_monthly | int | 20 |
| app_usage_days_monthly | int | 15 |
| research_reports_viewed | int | 5 |
| webinars_attended | int | 2 |
| sip_count | int | 3 |
| equity_holdings_count | int | 12 |
| ipo_applications | int | 4 |
| customer_service_calls | int | 1 |
| margin_utilisation_pct | float | 45.0 |
| referral_count | int | 2 |

---

## Output Columns

| Column | Description |
|--------|-------------|
| `propensity_score` | Ensemble probability (0–1). Higher = more likely to buy. |
| `xgb_score` | XGBoost individual score |
| `lgb_score` | LightGBM individual score |
| `propensity_segment` | HIGH (≥0.65) / MEDIUM (0.35–0.65) / LOW (<0.35) |
| `rank` | 1 = most likely buyer in the scored batch |

---

## Reading the Results

**HIGH** - Contact immediately. Strong buying signals across multiple features.

**MEDIUM** - Nurture. Send SIP calculators, research reports, educational content.

**LOW** - Skip for now. Route to generic brand awareness campaigns only.

The decile table in `reports/` is the most business-useful output. It shows conversion rate per score band. Typically the top 2 deciles (top 20% of customers by score) capture 50–60% of all actual buyers — meaning roughly 2x the conversion rate for the same outreach budget compared to random calling.

---

## Using This With Real Data

1. Export your customer base from your CRM/trading system as a CSV
2. Ensure column names match the Input Format table above
3. Label `will_buy_mutual_fund` in your historical data (who actually bought in the past 90 days)
4. Replace `generate_dataset.py` with a script that loads your real CSV
5. Run `train_model.py` to train on real data
6. Adjust `HIGH_THRESHOLD` and `MEDIUM_THRESHOLD` in `predict.py` to suit your campaign economics
7. Retrain monthly as customer behaviour shifts

---

## Deploying on AWS

**Batch scoring (recommended for propensity models):**
1. Push Docker image to AWS ECR
2. Set up an ECS scheduled task triggered weekly by EventBridge
3. Container reads customer data from S3, scores, writes results back to S3
4. Sales CRM reads the scored output every Monday morning

**Real-time scoring:**
1. Wrap `predict.py` in a FastAPI endpoint
2. Deploy on AWS ECS or SageMaker
3. POST a customer profile → receive a propensity score in response

---

## Tech Stack

| Component | Library |
|-----------|---------|
| ML Models | XGBoost 2.x, LightGBM 4.x |
| Data | Pandas, NumPy |
| Preprocessing | Scikit-learn |
| Class balancing | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn |
| Model saving | Joblib |
| Containerisation | Docker, Docker Compose |

---

## Author

**Vishal Saxena**