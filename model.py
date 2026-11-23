"""
MISE — v1 Regression Model
Business question: How much revenue will this bakery generate in a given hour,
so the baker knows how much to prepare before that period starts?

Data: French bakery POS data, Jan 2021–Sep 2022, 234k item rows
      (source: Kaggle "Bakery Sales" dataset)
Weather: Open-Meteo historical archive, Paris (lat=48.8566, lon=2.3522)
         Location assumed — dataset does not state city [assumption]
"""

import pathlib
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load & aggregate ──────────────────────────────────────────────────────

DATA_PATH    = pathlib.Path(__file__).parent / "bakery_sales.csv"
WEATHER_PATH = pathlib.Path(__file__).parent / "weather_paris.csv"


def parse_price(s: str) -> float:
    """'1,20 €'  →  1.20"""
    return float(str(s).replace("\xa0", "").replace("€", "").replace(",", ".").strip())


def load_hourly(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["unit_price"] = df["unit_price"].apply(parse_price)
    df["revenue"]    = df["Quantity"] * df["unit_price"]
    df["hour"]       = pd.to_datetime(df["time"], format="%H:%M").dt.hour

    hourly = (
        df.groupby(["date", "hour"])
        .agg(
            revenue=("revenue", "sum"),
            transactions=("ticket_number", "nunique"),
            items=("Quantity", "sum"),
        )
        .reset_index()
    )
    hourly["day_of_week"] = hourly["date"].dt.dayofweek   # 0=Mon, 6=Sun
    hourly["month"]       = hourly["date"].dt.month
    hourly["is_weekend"]  = (hourly["day_of_week"] >= 5).astype(int)

    # hour dummies (drop hour=6 as reference — earliest opening hour)
    hour_dummies = pd.get_dummies(hourly["hour"], prefix="h", drop_first=False)
    hour_dummies = hour_dummies.drop(columns=["h_6"], errors="ignore")
    hourly = pd.concat([hourly, hour_dummies], axis=1)

    # weekend × hour interaction for each hour dummy
    for col in [c for c in hourly.columns if c.startswith("h_")]:
        hourly[f"{col}_wknd"] = hourly[col] * hourly["is_weekend"]

    return hourly


# ── 2. Fetch / cache weather ──────────────────────────────────────────────────

def fetch_weather(start: str, end: str, cache: pathlib.Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["datetime"])

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude=48.8566&longitude=2.3522"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=Europe%2FParis"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()["hourly"]
    wx = pd.DataFrame({
        "datetime":  pd.to_datetime(data["time"]),
        "temp_c":    data["temperature_2m"],
        "precip_mm": data["precipitation"],
        "wind_kmh":  data["wind_speed_10m"],
    })
    wx["date"] = wx["datetime"].dt.normalize()
    wx["hour"] = wx["datetime"].dt.hour
    wx.to_csv(cache, index=False)
    print(f"Weather fetched and cached to {cache.name}")
    return wx


# ── 3. Merge ──────────────────────────────────────────────────────────────────

def merge_weather(hourly: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    wx["date"] = pd.to_datetime(wx["date"])
    merged = hourly.merge(
        wx[["date", "hour", "temp_c", "precip_mm"]],
        on=["date", "hour"],
        how="left",
    )
    merged["is_raining"] = (merged["precip_mm"] > 0.1).astype(int)
    merged.dropna(inplace=True)
    return merged


# ── 4. Build feature list ─────────────────────────────────────────────────────

CALENDAR_FEATURES = ["day_of_week", "month", "is_weekend", "temp_c", "precip_mm", "is_raining"]
OUTCOME = "revenue"


def get_features(df: pd.DataFrame) -> list[str]:
    hour_cols = [c for c in df.columns if c.startswith("h_") and not c.endswith("_wknd")]
    wknd_cols = [c for c in df.columns if c.endswith("_wknd")]
    return CALENDAR_FEATURES + hour_cols + wknd_cols


# ── 5. Split ──────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame, test_size: float = 0.30, seed: int = 42):
    return train_test_split(df, test_size=test_size, random_state=seed)


# ── 6. Fit ────────────────────────────────────────────────────────────────────

def fit(train: pd.DataFrame, features: list[str]) -> LinearRegression:
    model = LinearRegression()
    model.fit(train[features], train[OUTCOME])
    return model


def adjusted_r2(r2: float, n: int, k: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


# ── 7. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(model, train, test, features):
    y_train_pred = model.predict(train[features])
    r2_train = r2_score(train[OUTCOME], y_train_pred)
    adj_r2   = adjusted_r2(r2_train, len(train), len(features))

    y_test_pred   = model.predict(test[features])
    r2_test       = r2_score(test[OUTCOME], y_test_pred)
    rmse          = np.sqrt(mean_squared_error(test[OUTCOME], y_test_pred))

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
        "y_true":        test[OUTCOME].values,
        "y_pred":        y_test_pred,
        "y_baseline":    baseline_pred,
    }


# ── 8. Decision rule ──────────────────────────────────────────────────────────

def predict_and_decide(model, df_template: pd.DataFrame, features: list[str],
                        hour: int, day_of_week: int, month: int, is_weekend: int,
                        temp_c: float, precip_mm: float) -> dict:
    """
    Takes one hour-slot input and returns expected revenue + prep recommendation.

    Rule (thresholds based on training mean ± 0.5 std):
      predicted_revenue >= HIGH_THRESH  → FULL batch
      predicted_revenue >= LOW_THRESH   → HALF batch
      otherwise                         → MINIMAL batch
    """
    is_raining = int(precip_mm > 0.1)

    # build a one-row input matching all feature columns
    row = {f: 0 for f in features}
    row.update({
        "day_of_week": day_of_week,
        "month":       month,
        "is_weekend":  is_weekend,
        "temp_c":      temp_c,
        "precip_mm":   precip_mm,
        "is_raining":  is_raining,
    })
    h_col = f"h_{hour}"
    if h_col in row:
        row[h_col] = 1
    wknd_col = f"h_{hour}_wknd"
    if wknd_col in row:
        row[wknd_col] = is_weekend

    x = pd.DataFrame([row])
    pred = max(0.0, round(float(model.predict(x[features])[0]), 2))

    mean = df_template[OUTCOME].mean()
    std  = df_template[OUTCOME].std()
    high = mean + 0.5 * std
    low  = mean - 0.5 * std

    if pred >= high:
        action = "FULL batch"
    elif pred >= low:
        action = "HALF batch"
    else:
        action = "MINIMAL batch"

    return {"predicted_revenue_eur": pred, "bake_recommendation": action}


# ── 9. Value quantification ───────────────────────────────────────────────────

def quantify_value(metrics: dict, train: pd.DataFrame) -> None:
    """
    Assumptions (labelled):
      • Over-prediction (bake too much) → waste cost = 25% of error  [assumption]
      • Under-prediction (sell out)     → stockout cost = 30% of error [assumption]
      • ~200 baking decisions per month                               [assumption]
    Rates applied row-by-row: over-predictions use 25%, under-predictions use 30%.
    """
    waste_rate   = 0.25
    miss_rate    = 0.30
    decisions_mo = 200

    def row_cost(y_true, y_pred):
        errors = y_pred - y_true
        cost = np.where(errors > 0,
                        errors * waste_rate,
                        np.abs(errors) * miss_rate)
        return cost.mean()

    baseline_cost_per = row_cost(metrics["y_true"], metrics["y_baseline"])
    v1_cost_per       = row_cost(metrics["y_true"], metrics["y_pred"])

    baseline_cost_mo  = baseline_cost_per * decisions_mo
    v1_cost_mo        = v1_cost_per       * decisions_mo
    saving_mo         = baseline_cost_mo  - v1_cost_mo

    print("\n── Value quantification (assumptions labelled) ──────────────────")
    print(f"  Avg hourly revenue (training):        €{metrics['mean_train']:.2f}")
    print(f"  Assumed waste cost rate (over):       {waste_rate*100:.0f}%  [assumption]")
    print(f"  Assumed stockout cost rate (under):   {miss_rate*100:.0f}%  [assumption]")
    print(f"  Assumed bake decisions/month:         {decisions_mo}  [assumption]")
    print(f"  Avg cost per decision — baseline:     €{baseline_cost_per:.2f}")
    print(f"  Avg cost per decision — v1:           €{v1_cost_per:.2f}")
    print(f"  Estimated monthly cost — baseline:    €{baseline_cost_mo:.0f}")
    print(f"  Estimated monthly cost — v1:          €{v1_cost_mo:.0f}")
    print(f"  Estimated monthly saving from v1:     €{saving_mo:.0f}")


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_hourly(DATA_PATH)
    print(f"Loaded {len(df)} hourly buckets  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    wx = fetch_weather("2021-01-02", "2022-09-30", WEATHER_PATH)
    df = merge_weather(df, wx)
    features = get_features(df)
    print(f"After weather merge: {len(df)} rows  |  {len(features)} features")

    train, test = split(df)
    print(f"Train: {len(train)}  |  Test: {len(test)}")

    model = fit(train, features)

    metrics = evaluate(model, train, test, features)
    print("\n── Fit evaluation ──────────────────────────────────────────────")
    print(f"  Adjusted R² (training):               {metrics['adj_r2_train']:.4f}")
    print(f"  R² baseline (test):                   {metrics['r2_baseline']:.4f}")
    print(f"  R² v1 (test):                         {metrics['r2_test']:.4f}")
    print(f"  RMSE baseline (test, €):              €{metrics['rmse_baseline']:.2f}")
    print(f"  RMSE v1 (test, €):                    €{metrics['rmse_test']:.2f}")

    quantify_value(metrics, train)

    print("\n── Decision for new input ──────────────────────────────────────")
    example = dict(hour=10, day_of_week=5, month=12,
                   is_weekend=1, temp_c=4.0, precip_mm=2.5)
    result = predict_and_decide(model, train, features, **example)
    print(f"  Input: {example}")
    print(f"  Predicted revenue:  €{result['predicted_revenue_eur']}")
    print(f"  Recommendation:     {result['bake_recommendation']}")

    return model, metrics


if __name__ == "__main__":
    main()
