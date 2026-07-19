# 08_rolling_baseline.py
# PURPOSE: Build the rolling revenue baseline per category per week.
# OUTPUT: The deviation from this baseline IS your anomaly signal.
# Phase 3 will read this file directly.

import pandas as pd
import numpy as np
import os

os.makedirs("data/outputs", exist_ok=True)

# Load the category × week data we built in Step 4
cat_weekly = pd.read_csv("data/outputs/trend_02_category_weekly.csv")
cat_weekly = cat_weekly.sort_values(["category_en", "year_week"])

# ── Rolling baseline per category ────────────────────────────
# For each category, for each week:
#   - rolling_mean   = average revenue of past 8 weeks
#   - rolling_std    = std deviation of past 8 weeks
#   - rolling_median = median of past 8 weeks (more robust to spikes)
#   - upper_bound    = mean + 2*std  (normal ceiling)
#   - lower_bound    = mean - 2*std  (normal floor)
#   - z_score        = how many std devs this week is from the mean

def add_baseline(group):
    group = group.sort_values("year_week").copy()

    # min_periods=4 means we need at least 4 weeks of history
    # before we start flagging anomalies
    group["rolling_mean"]   = (group["weekly_revenue"]
                                .shift(1)           # don't include current week
                                .rolling(8, min_periods=4)
                                .mean())

    group["rolling_std"]    = (group["weekly_revenue"]
                                .shift(1)
                                .rolling(8, min_periods=4)
                                .std())

    group["rolling_median"] = (group["weekly_revenue"]
                                .shift(1)
                                .rolling(8, min_periods=4)
                                .median())

    group["upper_bound"]    = group["rolling_mean"] + 2 * group["rolling_std"]
    group["lower_bound"]    = (group["rolling_mean"] - 2 * group["rolling_std"]).clip(lower=0)

    # Z-score: how far is this week from "normal"?
    group["z_score"]        = ((group["weekly_revenue"] - group["rolling_mean"])
                                / group["rolling_std"].replace(0, np.nan))

    # Flag: True if this week is outside the normal band
    group["is_anomaly"]     = (
        (group["weekly_revenue"] < group["lower_bound"]) |
        (group["weekly_revenue"] > group["upper_bound"])
    )

    # Revenue at risk = how much below the lower bound is this week?
    group["revenue_at_risk"] = (
        (group["lower_bound"] - group["weekly_revenue"])
        .clip(lower=0)
        .round(2)
    )

    return group

# Apply baseline to every category separately
baseline_df = (
    cat_weekly
    .groupby("category_en", group_keys=False)
    .apply(add_baseline)
    .round(2)
)

# ── Summary: Which categories have the most anomalies? ─────────
anomaly_summary = (
    baseline_df[baseline_df["is_anomaly"] == True]
    .groupby("category_en")
    .agg(
        anomaly_weeks   =("is_anomaly", "sum"),
        total_at_risk   =("revenue_at_risk", "sum"),
        avg_z_score     =("z_score", "mean")
    )
    .round(2)
    .sort_values("total_at_risk", ascending=False)
    .reset_index()
    .head(15)
)

print("=== TOP CATEGORIES BY REVENUE AT RISK ===")
print(anomaly_summary.to_string(index=False))

# ── Save outputs ───────────────────────────────────────────────
baseline_df.to_csv("data/outputs/baseline_category_weekly.csv", index=False)
anomaly_summary.to_csv("data/outputs/baseline_anomaly_summary.csv", index=False)

print(f"\nBaseline built for {baseline_df['category_en'].nunique()} categories")
print(f"Total anomaly flags (Z-score): {baseline_df['is_anomaly'].sum()}")
print(f"Total revenue at risk identified: R${baseline_df['revenue_at_risk'].sum():,.0f}")
print("\nSaved:")
print("  data/outputs/baseline_category_weekly.csv")
print("  data/outputs/baseline_anomaly_summary.csv")