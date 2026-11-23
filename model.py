"""
MISE — v1 Regression Model
Business question: How many transactions will occur in a given hour on a given day,
so the baker knows how much to prepare before that period starts?

Weather data: Open-Meteo historical archive (free, no API key)
  latitude=55.9533, longitude=-3.1883 (Edinburgh, UK)
  Variables: temperature_2m (°C), precipitation (mm), wind_speed_10m (km/h)
"""

import sys
import pathlib
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load & aggregate transactions ────────────────────────────────────────

DATA_PATH    = pathlib.Path(__file__).parent / "BreadBasket_DMS.csv"
WEATHER_PATH = pathlib.Path(__file__).parent / "weather_edinburgh.csv"

def load_hourly(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    hourly = (
        df.groupby(["Date", "hour"])["Transaction"]
        .nunique()
        .reset_index()
    )
    hourly.columns = ["date", "hour", "transactions"]
    hourly["day_of_week"] = hourly["date"].dt.dayofweek
    hourly["month"]       = hourly["date"].dt.month
    hourly["is_weekend"]  = (hourly["day_of_week"] >= 5).astype(int)
    return hourly


# ── 2. Fetch / cache weather ─────────────────────────────────────────────────

def fetch_weather(start: str, end: str, cache: pathlib.Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["datetime"])

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude=55.9533&longitude=-3.1883"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=Europe%2FLondon"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    wx = pd.DataFrame({
        "datetime":      pd.to_datetime(data["time"]),
        "temp_c":        data["temperature_2m"],
        "precip_mm":     data["precipitation"],
        "wind_kmh":      data["wind_speed_10m"],
    })
    wx["date"] = wx["datetime"].dt.date
    wx["hour"] = wx["datetime"].dt.hour
    wx.to_csv(cache, index=False)
    print(f"Weather fetched and cached to {cache.name}")
    return wx


# ── 3. Merge ──────────────────────────────────────────────────────────────────

def merge_weather(hourly: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    wx["date"] = pd.to_datetime(wx["date"])
    merged = hourly.merge(
        wx[["date", "hour", "temp_c", "precip_mm", "wind_kmh"]],
        on=["date", "hour"],
        how="left",
    )
    merged["is_raining"] = (merged["precip_mm"] > 0.1).astype(int)
    merged.dropna(inplace=True)
    return merged


# ── 4. Features / outcome ─────────────────────────────────────────────────────

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temp_c", "precip_mm", "is_raining",
]
OUTCOME = "transactions"


# ── 5. Split ──────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame, test_size: float = 0.30, seed: int = 42):
    return train_test_split(df, test_size=test_size, random_state=seed)


# ── 6. Fit ────────────────────────────────────────────────────────────────────

def fit(train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(train[FEATURES], train[OUTCOME])
    return model


def adjusted_r2(r2: float, n: int, k: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


# ── 7. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(model: LinearRegression, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    y_train_pred = model.predict(train[FEATURES])
    r2_train = r2_score(train[OUTCOME], y_train_pred)
    adj_r2   = adjusted_r2(r2_train, len(train), len(FEATURES))

    y_test_pred = model.predict(test[FEATURES])
    r2_test  = r2_score(test[OUTCOME], y_test_pred)
    rmse     = np.sqrt(mean_squared_error(test[OUTCOME], y_test_pred))

    baseline_pred = np.full(len(test), train[OUTCOME].mean())
    r2_baseline   = r2_score(test[OUTCOME], baseline_pred)
    rmse_baseline = np.sqrt(mean_squared_error(test[OUTCOME], baseline_pred))

    return {
        "adj_r2_train":  adj_r2,
        "r2_test":       r2_test,
        "rmse_test":     rmse,
        "r2_baseline":   r2_baseline,
        "rmse_baseline": rmse_baseline,
        "mean_train":    train[OUTCOME].mean(),
    }


# ── 8. Decision rule ──────────────────────────────────────────────────────────

def predict_and_decide(model: LinearRegression, hour: int, day_of_week: int,
                        month: int, is_weekend: int,
                        temp_c: float, precip_mm: float) -> dict:
    """
    Takes one hour-slot input and returns a bake recommendation.

    Rule: predicted_transactions >= 10 → FULL batch
          5–9                           → HALF batch
          < 5                           → MINIMAL batch
    """
    is_raining = int(precip_mm > 0.1)
    x = pd.DataFrame([{
        "hour":        hour,
        "day_of_week": day_of_week,
        "month":       month,
        "is_weekend":  is_weekend,
        "temp_c":      temp_c,
        "precip_mm":   precip_mm,
        "is_raining":  is_raining,
    }])
    pred = max(0, round(model.predict(x)[0], 1))

    if pred >= 10:
        action = "FULL batch"
    elif pred >= 5:
        action = "HALF batch"
    else:
        action = "MINIMAL batch"

    return {"predicted_transactions": pred, "bake_recommendation": action}


# ── 9. Value quantification ───────────────────────────────────────────────────

def quantify_value(metrics: dict) -> None:
    """
    Assumptions (labelled — no real cost figures available):
      • Each wasted item costs ~£0.80 (ingredient + labour)
      • A missed sale (stockout) costs ~£1.50 in lost margin
      • ~200 unique baking decisions per month (opening hours × 25 days)
    """
    rmse_base    = metrics["rmse_baseline"]
    rmse_v1      = metrics["rmse_test"]
    waste_cost   = 0.80
    miss_cost    = 1.50
    decisions_mo = 200

    baseline_cost_mo = (rmse_base * (waste_cost + miss_cost) / 2) * decisions_mo
    v1_cost_mo       = (rmse_v1   * (waste_cost + miss_cost) / 2) * decisions_mo
    saving_mo        = baseline_cost_mo - v1_cost_mo

    print("\n── Value quantification (assumptions labelled) ──────────────────")
    print(f"  Avg transactions/hour (training):     {metrics['mean_train']:.1f}")
    print(f"  Baseline RMSE (test):                 {rmse_base:.2f} tx")
    print(f"  v1 RMSE (test):                       {rmse_v1:.2f} tx")
    print(f"  Assumed waste cost per item:          £{waste_cost:.2f}  [assumption]")
    print(f"  Assumed missed-sale cost per item:    £{miss_cost:.2f}  [assumption]")
    print(f"  Assumed bake decisions/month:         {decisions_mo}  [assumption]")
    print(f"  Estimated monthly cost — baseline:    £{baseline_cost_mo:.0f}")
    print(f"  Estimated monthly cost — v1:          £{v1_cost_mo:.0f}")
    print(f"  Estimated monthly saving from v1:     £{saving_mo:.0f}")


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_hourly(DATA_PATH)
    print(f"Loaded {len(df)} hourly buckets  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    wx = fetch_weather("2016-10-30", "2017-04-09", WEATHER_PATH)
    df = merge_weather(df, wx)
    print(f"After weather merge: {len(df)} rows  |  features: {FEATURES}")

    train, test = split(df)
    print(f"Train: {len(train)}  |  Test: {len(test)}")

    model = fit(train)
    print("\n── Model coefficients ──────────────────────────────────────────")
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"  {feat:<15} {coef:+.4f}")
    print(f"  intercept       {model.intercept_:+.4f}")

    metrics = evaluate(model, train, test)
    print("\n── Fit evaluation ──────────────────────────────────────────────")
    print(f"  Adjusted R² (training):               {metrics['adj_r2_train']:.4f}")
    print(f"  R² baseline (test):                   {metrics['r2_baseline']:.4f}")
    print(f"  R² v1 (test):                         {metrics['r2_test']:.4f}")
    print(f"  RMSE baseline (test, transactions):   {metrics['rmse_baseline']:.2f}")
    print(f"  RMSE v1 (test, transactions):         {metrics['rmse_test']:.2f}")

    quantify_value(metrics)

    print("\n── Decision for new input ──────────────────────────────────────")
    example = dict(hour=10, day_of_week=5, month=12,
                   is_weekend=1, temp_c=4.0, precip_mm=2.5)
    result = predict_and_decide(model, **example)
    print(f"  Input: {example}")
    print(f"  Predicted transactions: {result['predicted_transactions']}")
    print(f"  Recommendation:         {result['bake_recommendation']}")

    return model, metrics


if __name__ == "__main__":
    main()
