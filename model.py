"""
MISE — v1 Regression Model
Business question: How many transactions will occur in a given hour on a given day,
so the baker knows how much to prepare before that period starts?
"""

import sys
import pathlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load & aggregate ──────────────────────────────────────────────────────

DATA_PATH = pathlib.Path(__file__).parent / "BreadBasket_DMS.csv"

def load_hourly(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    # one row per (date, hour) → count unique transactions
    hourly = (
        df.groupby(["Date", "hour"])["Transaction"]
        .nunique()
        .reset_index()
    )
    hourly.columns = ["date", "hour", "transactions"]
    hourly["day_of_week"] = hourly["date"].dt.dayofweek   # 0=Mon, 6=Sun
    hourly["month"]       = hourly["date"].dt.month
    hourly["is_weekend"]  = (hourly["day_of_week"] >= 5).astype(int)
    return hourly


# ── 2. Features / outcome ────────────────────────────────────────────────────

FEATURES = ["hour", "day_of_week", "month", "is_weekend"]
OUTCOME  = "transactions"


# ── 3. Split ─────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame, test_size: float = 0.30, seed: int = 42):
    return train_test_split(df, test_size=test_size, random_state=seed)


# ── 4. Fit ───────────────────────────────────────────────────────────────────

def fit(train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(train[FEATURES], train[OUTCOME])
    return model


def adjusted_r2(r2: float, n: int, k: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


# ── 5. Evaluate ──────────────────────────────────────────────────────────────

def evaluate(model: LinearRegression, train: pd.DataFrame, test: pd.DataFrame):
    # training adjusted R²
    y_train_pred = model.predict(train[FEATURES])
    r2_train = r2_score(train[OUTCOME], y_train_pred)
    adj_r2   = adjusted_r2(r2_train, len(train), len(FEATURES))

    # test R² and RMSE (typical error in transactions)
    y_test_pred = model.predict(test[FEATURES])
    r2_test  = r2_score(test[OUTCOME], y_test_pred)
    rmse     = np.sqrt(mean_squared_error(test[OUTCOME], y_test_pred))

    # baseline: predict mean of training outcome for every test row
    baseline_pred = np.full(len(test), train[OUTCOME].mean())
    r2_baseline   = r2_score(test[OUTCOME], baseline_pred)
    rmse_baseline = np.sqrt(mean_squared_error(test[OUTCOME], baseline_pred))

    return {
        "adj_r2_train": adj_r2,
        "r2_test":      r2_test,
        "rmse_test":    rmse,
        "r2_baseline":  r2_baseline,
        "rmse_baseline": rmse_baseline,
        "mean_train":   train[OUTCOME].mean(),
    }


# ── 6. Decision rule ─────────────────────────────────────────────────────────

def predict_and_decide(model: LinearRegression, hour: int, day_of_week: int,
                        month: int, is_weekend: int) -> dict:
    """
    Takes one hour-slot input your app will see.
    Returns predicted transaction count and a bake recommendation.

    Rule: if predicted_transactions >= 10, bake a full batch (HIGH);
          if 5–9, bake a half batch (MEDIUM);
          otherwise bake minimal (LOW).
    """
    x = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
    }])
    pred = max(0, round(model.predict(x)[0], 1))

    if pred >= 10:
        action = "FULL batch"
    elif pred >= 5:
        action = "HALF batch"
    else:
        action = "MINIMAL batch"

    return {"predicted_transactions": pred, "bake_recommendation": action}


# ── 7. Value quantification ──────────────────────────────────────────────────

def quantify_value(metrics: dict) -> None:
    """
    Assumptions (labelled — no real cost figures available):
      • Each wasted item costs ~£0.80 (ingredient + labour)
      • A missed sale (stockout) costs ~£1.50 in lost margin
      • Baseline over-bakes by ~20% and under-bakes by ~10% of hourly demand
      • ~200 unique baking decisions per month (opening hours × 25 days)
    """
    avg_tx       = metrics["mean_train"]           # avg transactions/hour
    rmse_base    = metrics["rmse_baseline"]        # baseline error (transactions)
    rmse_v1      = metrics["rmse_test"]            # v1 error

    waste_cost   = 0.80    # £ per wasted item (assumption)
    miss_cost    = 1.50    # £ per missed sale  (assumption)
    decisions_mo = 200     # bake decisions per month (assumption)

    # each transaction error → roughly one over/under-baked item
    baseline_cost_mo = (rmse_base * (waste_cost + miss_cost) / 2) * decisions_mo
    v1_cost_mo       = (rmse_v1   * (waste_cost + miss_cost) / 2) * decisions_mo
    saving_mo        = baseline_cost_mo - v1_cost_mo

    print("\n── Value quantification (assumptions labelled) ──────────────────")
    print(f"  Avg transactions/hour (training):     {avg_tx:.1f}")
    print(f"  Baseline RMSE (test):                 {rmse_base:.2f} tx")
    print(f"  v1 RMSE (test):                       {rmse_v1:.2f} tx")
    print(f"  Assumed waste cost per item:          £{waste_cost:.2f}  [assumption]")
    print(f"  Assumed missed-sale cost per item:    £{miss_cost:.2f}  [assumption]")
    print(f"  Assumed bake decisions/month:         {decisions_mo}  [assumption]")
    print(f"  Estimated monthly cost — baseline:    £{baseline_cost_mo:.0f}")
    print(f"  Estimated monthly cost — v1:          £{v1_cost_mo:.0f}")
    print(f"  Estimated monthly saving from v1:     £{saving_mo:.0f}")


# ── 8. Main ──────────────────────────────────────────────────────────────────

def main():
    # load
    df = load_hourly(DATA_PATH)
    print(f"Loaded {len(df)} hourly buckets from {df['date'].min().date()} "
          f"to {df['date'].max().date()}")
    print(f"Rows: {len(df)}  |  70/30 split  |  features: {FEATURES}")

    # split
    train, test = split(df)
    print(f"Train: {len(train)}  |  Test: {len(test)}")

    # fit
    model = fit(train)
    print("\n── Model coefficients ──────────────────────────────────────────")
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"  {feat:<15} {coef:+.4f}")
    print(f"  intercept       {model.intercept_:+.4f}")

    # evaluate
    metrics = evaluate(model, train, test)
    print("\n── Fit evaluation ──────────────────────────────────────────────")
    print(f"  Adjusted R² (training):               {metrics['adj_r2_train']:.4f}")
    print(f"  R² baseline (test):                   {metrics['r2_baseline']:.4f}")
    print(f"  R² v1 (test):                         {metrics['r2_test']:.4f}")
    print(f"  RMSE baseline (test, transactions):   {metrics['rmse_baseline']:.2f}")
    print(f"  RMSE v1 (test, transactions):         {metrics['rmse_test']:.2f}")

    # value
    quantify_value(metrics)

    # decision for a new input
    print("\n── Decision for new input ──────────────────────────────────────")
    example = dict(hour=10, day_of_week=5, month=12, is_weekend=1)  # Sat 10am Dec
    result  = predict_and_decide(model, **example)
    print(f"  Input: {example}")
    print(f"  Predicted transactions: {result['predicted_transactions']}")
    print(f"  Recommendation:         {result['bake_recommendation']}")

    return model, metrics


if __name__ == "__main__":
    main()
