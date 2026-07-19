# 09_zscore_detector.py
# PURPOSE: Detect anomalous revenue weeks using Z-score method.
# Simple, fast, and fully explainable to non-technical stakeholders.

import pandas as pd
import numpy as np
import os

os.makedirs("data/outputs/anomalies", exist_ok=True)

# Load the rolling baseline we built in Phase 2
baseline = pd.read_csv("data/outputs/baseline_category_weekly.csv")

# ── Z-score detection ─────────────────────────────────────────
# We already computed z_score in Phase 2 baseline.
# Now we apply business rules on top of it.

# Rule 1: Z-score below -2 = revenue significantly lower than normal
# Rule 2: Must have at least 4 weeks of history (rolling_mean not null)
# Rule 3: Only flag categories with meaningful revenue (avoid noise
#          from tiny categories with 2 orders/week)

MIN_BASELINE_REVENUE = 500   # minimum avg weekly revenue to be worth flagging
ZSCORE_THRESHOLD     = -2.0  # standard deviations below mean

anomalies_zscore = baseline[
    (baseline["z_score"] < ZSCORE_THRESHOLD) &
    (baseline["rolling_mean"] >= MIN_BASELINE_REVENUE) &
    (baseline["rolling_mean"].notna())
].copy()

# ── Enrich each anomaly with business context ─────────────────
anomalies_zscore["severity"] = pd.cut(
    anomalies_zscore["z_score"],
    bins=[-np.inf, -3.0, -2.5, -2.0],
    labels=["Critical", "High", "Medium"]
)

anomalies_zscore["expected_revenue"] = anomalies_zscore["rolling_mean"].round(2)
anomalies_zscore["actual_revenue"]   = anomalies_zscore["weekly_revenue"].round(2)
anomalies_zscore["revenue_gap"]      = (
    anomalies_zscore["expected_revenue"] - anomalies_zscore["actual_revenue"]
).round(2)

anomalies_zscore["revenue_gap_pct"]  = (
    100 * anomalies_zscore["revenue_gap"] / anomalies_zscore["expected_revenue"]
).round(1)

# ── Summary ───────────────────────────────────────────────────
print("=== Z-SCORE ANOMALY DETECTION RESULTS ===\n")
print(f"Total anomalies flagged : {len(anomalies_zscore)}")
print(f"Categories affected     : {anomalies_zscore['category_en'].nunique()}")
print(f"Total revenue gap       : R${anomalies_zscore['revenue_gap'].sum():,.0f}")

print("\nBy severity:")
print(anomalies_zscore["severity"].value_counts().to_string())

print("\nTop 10 anomalies by revenue gap:")
top10 = (anomalies_zscore
         .sort_values("revenue_gap", ascending=False)
         [["year_week","category_en","actual_revenue",
           "expected_revenue","revenue_gap","z_score","severity"]]
         .head(10))
print(top10.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────
anomalies_zscore.to_csv(
    "data/outputs/anomalies/zscore_anomalies.csv", index=False
)
print("\nSaved → data/outputs/anomalies/zscore_anomalies.csv")