# 10_isolation_forest.py
# PURPOSE: Multi-dimensional anomaly detection using Isolation Forest.
# Catches compound anomalies that Z-score misses.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("data/outputs/anomalies", exist_ok=True)

# Load data
baseline  = pd.read_csv("data/outputs/baseline_category_weekly.csv")
cat_weekly = pd.read_csv("data/outputs/trend_02_category_weekly.csv")

# Merge to get all features in one place
data = baseline.merge(
    cat_weekly[["year_week","category_en","cancellations","avg_order_value"]],
    on=["year_week","category_en"],
    how="left"
)
# print(data.columns.tolist())

# Fix duplicate columns
data["cancellations"] = data["cancellations_x"].fillna(data["cancellations_y"])
data["avg_order_value"] = data["avg_order_value_x"].fillna(data["avg_order_value_y"])

# Clean up
data.drop(columns=[
    "cancellations_x", "cancellations_y",
    "avg_order_value_x", "avg_order_value_y"
], inplace=True)

# ── Feature engineering ───────────────────────────────────────
# These are the features Isolation Forest will use.
# Each one captures a different dimension of revenue health.

data["cancellation_rate"] = (
    data["cancellations"] / data["orders_count"].replace(0, np.nan)
).fillna(0)

data["revenue_vs_baseline"] = (
    data["weekly_revenue"] / data["rolling_mean"].replace(0, np.nan)
).fillna(1)  # ratio of actual to expected (1.0 = exactly normal)

data["orders_vs_baseline"] = (
    data["orders_count"] /
    data["orders_count"].shift(1).rolling(8, min_periods=4).mean()
).fillna(1)

# Fill any remaining nulls with neutral values
data = data.fillna({
    "z_score":            0,
    "cancellation_rate":  0,
    "revenue_vs_baseline":1,
    "orders_vs_baseline": 1,
    "avg_order_value":    data["avg_order_value"].median()
})

# ── Train Isolation Forest per category ───────────────────────
# WHY per category? Because "normal" is different for each category.
# A 30% revenue drop in electronics is very different from
# a 30% drop in a tiny niche category.

FEATURES = [
    "weekly_revenue",       # raw revenue this week
    "revenue_vs_baseline",  # ratio to rolling baseline
    "cancellation_rate",    # % of orders cancelled
    "avg_order_value",      # average spend per order
    "z_score",              # statistical deviation
    "orders_count",         # volume of orders
]

results = []

# Only run on categories with enough data (at least 12 weeks)
category_counts = data.groupby("category_en")["year_week"].count()
valid_categories = category_counts[category_counts >= 12].index

for category in valid_categories:
    cat_data = data[data["category_en"] == category].copy()
    X = cat_data[FEATURES].values

    # Normalize features — important so revenue (large numbers)
    # doesn't dominate over cancellation_rate (small numbers)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination = expected % of anomalies in the data
    # We set 0.05 = we expect ~5% of weeks to be anomalous
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    clf.fit(X_scaled)

    # Scores: more negative = more anomalous
    cat_data["if_score"]      = clf.score_samples(X_scaled)
    # Predictions: -1 = anomaly, 1 = normal
    cat_data["if_prediction"] = clf.predict(X_scaled)
    cat_data["if_anomaly"]    = cat_data["if_prediction"] == -1

    results.append(cat_data)

if_results = pd.concat(results, ignore_index=True)

# ── Compare Z-score vs Isolation Forest flags ─────────────────
# Points flagged by BOTH detectors = highest confidence anomalies
if_results["flagged_by_zscore"] = if_results["z_score"] < -2.0
if_results["flagged_by_both"]   = (
    if_results["if_anomaly"] & if_results["flagged_by_zscore"]
)

print("=== ISOLATION FOREST RESULTS ===\n")
print(f"Categories analysed     : {len(valid_categories)}")
print(f"Total IF anomalies      : {if_results['if_anomaly'].sum()}")
print(f"Total Z-score anomalies : {if_results['flagged_by_zscore'].sum()}")
print(f"Flagged by BOTH         : {if_results['flagged_by_both'].sum()}")
print("\n→ Anomalies flagged by both = highest confidence leakage signals\n")

# Top anomalies by IF score (most negative = most anomalous)
top_if = (if_results[if_results["if_anomaly"]]
          .sort_values("if_score")
          [["year_week","category_en","weekly_revenue",
            "revenue_vs_baseline","cancellation_rate",
            "if_score","flagged_by_both"]]
          .head(10))
print("Top 10 Isolation Forest anomalies:")
print(top_if.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────
if_results.to_csv("data/outputs/anomalies/isolation_forest_results.csv",
                  index=False)

# Save high-confidence anomalies (flagged by both) separately
both_flagged = if_results[if_results["flagged_by_both"]].copy()
both_flagged.to_csv("data/outputs/anomalies/high_confidence_anomalies.csv",
                    index=False)

print(f"\nSaved → isolation_forest_results.csv")
print(f"Saved → high_confidence_anomalies.csv ({len(both_flagged)} rows)")