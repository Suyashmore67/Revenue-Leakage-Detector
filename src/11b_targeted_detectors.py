# 11b_targeted_detectors.py
# PURPOSE: One dedicated detector per leakage type.
# Each detector measures the RIGHT signal for that leakage.
# This is how production anomaly systems actually work.

import pandas as pd
import numpy as np
import sqlite3
import os

os.makedirs("data/outputs/anomalies", exist_ok=True)

DB_PATH = "data/db/olist.db"

def run_query(sql):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)

ground_truth = pd.read_csv("data/processed/leakage_ground_truth.csv")

all_detections = []


# ══════════════════════════════════════════════════════════════
# DETECTOR 1: Discount Abuse (L1)
# Signal: payment_value is significantly lower than item list price
# We rebuild this using ORIGINAL payments vs item prices directly
# ══════════════════════════════════════════════════════════════

print("Running Detector 1: Discount Abuse...")

discount_sql = """
SELECT
    o.order_id,
    ROUND(SUM(i.price + i.freight_value), 2)   AS list_price,
    ROUND(SUM(p.payment_value), 2)              AS actual_paid,
    ROUND(
        100.0 * (SUM(i.price + i.freight_value) - SUM(p.payment_value))
              /  SUM(i.price + i.freight_value), 2
    )                                           AS discount_pct

FROM orders o
JOIN order_items i      ON o.order_id = i.order_id
JOIN order_payments p   ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
  AND p.payment_value > 0
GROUP BY o.order_id
HAVING discount_pct > 20   -- flag orders with >20% discount
"""

discount_detected = run_query(discount_sql)
discount_detected["leakage_type"] = "L1_discount_abuse"
discount_detected["detected"] = True

# Evaluate against ground truth
l1_truth = ground_truth[ground_truth["leakage_type"] == "L1_discount_abuse"]
l1_tp = len(set(discount_detected["order_id"]) & set(l1_truth["order_id"]))
l1_fp = len(discount_detected) - l1_tp
l1_fn = len(l1_truth) - l1_tp
l1_precision = l1_tp / len(discount_detected) if len(discount_detected) > 0 else 0
l1_recall    = l1_tp / len(l1_truth) if len(l1_truth) > 0 else 0

print(f"  Flags raised : {len(discount_detected)}")
print(f"  True pos     : {l1_tp} / {len(l1_truth)} injected")
print(f"  Precision    : {l1_precision:.1%}  |  Recall: {l1_recall:.1%}")

discount_detected.to_csv(
    "data/outputs/anomalies/detected_L1_discount.csv", index=False
)
all_detections.append({
    "detector": "L1_discount_abuse", "flags": len(discount_detected),
    "true_positives": l1_tp, "false_positives": l1_fp,
    "false_negatives": l1_fn,
    "precision": round(l1_precision, 3), "recall": round(l1_recall, 3)
})


# ══════════════════════════════════════════════════════════════
# DETECTOR 2: Refund Spike (L2)
# Signal: cancellation rate in a category-week spikes above
#         its own 8-week rolling average by more than 2 std devs
# ══════════════════════════════════════════════════════════════

print("\nRunning Detector 2: Refund Spike...")

refund_sql = """
SELECT
    STRFTIME('%Y-W%W', o.order_purchase_timestamp)              AS year_week,
    COALESCE(x.product_category_name_english, pr.product_category_name)     AS category_en,
    COUNT(DISTINCT o.order_id)                                  AS total_orders,
    SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END) AS cancellations,
    ROUND(
        100.0 * SUM(CASE WHEN o.order_status='canceled' THEN 1 ELSE 0 END)
              / COUNT(DISTINCT o.order_id), 2
    )                                                           AS cancel_rate_pct

FROM orders o
JOIN order_items i      ON o.order_id   = i.order_id
JOIN products pr        ON i.product_id = pr.product_id
LEFT JOIN category_xlat x ON pr.product_category_name = x.product_category_name
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY year_week, category_en
ORDER BY category_en, year_week
"""

refund_df = run_query(refund_sql)

# Build rolling baseline for cancellation rate per category
def flag_refund_spikes(group):
    group = group.sort_values("year_week").copy()
    group["rolling_cancel_mean"] = (group["cancel_rate_pct"]
                                    .shift(1).rolling(8, min_periods=4).mean())
    group["rolling_cancel_std"]  = (group["cancel_rate_pct"]
                                    .shift(1).rolling(8, min_periods=4).std())
    group["cancel_z_score"] = (
        (group["cancel_rate_pct"] - group["rolling_cancel_mean"])
        / group["rolling_cancel_std"].replace(0, np.nan)
    )
    group["is_refund_spike"] = group["cancel_z_score"] > 2.0
    return group

refund_flagged = (refund_df
                  .groupby("category_en", group_keys=False)
                  .apply(flag_refund_spikes))

spikes = refund_flagged[refund_flagged["is_refund_spike"] == True]

# Evaluate: join back to order level via ground truth
l2_truth = ground_truth[ground_truth["leakage_type"] == "L2_refund_spike"]
orders_df = pd.read_csv("data/processed/orders_with_leakage.csv",
                        parse_dates=["order_purchase_timestamp"])
orders_df["year_week"] = orders_df["order_purchase_timestamp"].dt.strftime("%Y-W%W")

# Which weeks were flagged?
flagged_weeks = set(spikes["year_week"])
truth_weeks   = set(
    orders_df[orders_df["order_id"].isin(l2_truth["order_id"])]["year_week"]
)

l2_tp = len(flagged_weeks & truth_weeks)
l2_fp = len(flagged_weeks - truth_weeks)
l2_fn = len(truth_weeks  - flagged_weeks)
l2_precision = l2_tp / len(flagged_weeks) if flagged_weeks else 0
l2_recall    = l2_tp / len(truth_weeks)   if truth_weeks   else 0

print(f"  Flags raised : {len(spikes)} category-weeks")
print(f"  True pos     : {l2_tp} / {len(truth_weeks)} injected weeks")
print(f"  Precision    : {l2_precision:.1%}  |  Recall: {l2_recall:.1%}")

spikes.to_csv("data/outputs/anomalies/detected_L2_refund.csv", index=False)
all_detections.append({
    "detector": "L2_refund_spike", "flags": len(spikes),
    "true_positives": l2_tp, "false_positives": l2_fp,
    "false_negatives": l2_fn,
    "precision": round(l2_precision, 3), "recall": round(l2_recall, 3)
})


# ══════════════════════════════════════════════════════════════
# DETECTOR 3: Silent Churn (L3)
# Signal: high-value customers with no orders in the last 6 months
# This is a CUSTOMER-level detector, not a revenue-level one
# ══════════════════════════════════════════════════════════════

print("\nRunning Detector 3: Silent Churn...")

# Step 1: Build customer LTV
ltv_sql = """
SELECT
    c.customer_unique_id,
    MAX(o.order_purchase_timestamp)             AS last_order_date,
    COUNT(DISTINCT o.order_id)                  AS total_orders,
    ROUND(SUM(p.payment_value), 2)              AS lifetime_value
FROM customers c
JOIN orders o        ON c.customer_id  = o.customer_id
JOIN order_payments p ON o.order_id   = p.order_id
WHERE p.payment_value > 0
GROUP BY c.customer_unique_id
"""
ltv_df = run_query(ltv_sql)
ltv_df["last_order_date"] = pd.to_datetime(ltv_df["last_order_date"])

# Step 2: Identify top 20% by LTV
top20_threshold = ltv_df["lifetime_value"].quantile(0.80)
ltv_df["is_high_value"] = ltv_df["lifetime_value"] >= top20_threshold

# Step 3: Find reference date (last date in dataset)
reference_date = ltv_df["last_order_date"].max()

# Step 4: Flag high-value customers inactive for 6+ months
ltv_df["days_inactive"] = (reference_date - ltv_df["last_order_date"]).dt.days
ltv_df["is_churned"]    = (
    ltv_df["is_high_value"] &
    (ltv_df["days_inactive"] >= 180)
)

churned = ltv_df[ltv_df["is_churned"]]

# Evaluate against ground truth
# L3 ground truth = order_ids that were removed from high-value customers
l3_truth = ground_truth[ground_truth["leakage_type"] == "L3_silent_churn"]

# Get the customer_unique_ids behind those order_ids
customers_raw = pd.read_csv("data/raw/olist_customers_dataset.csv")
orders_raw    = pd.read_csv("data/raw/olist_orders_dataset.csv")

truth_customers = set(
    orders_raw[orders_raw["order_id"].isin(l3_truth["order_id"])]
    .merge(customers_raw, on="customer_id")["customer_unique_id"]
)
detected_customers = set(churned["customer_unique_id"])

l3_tp = len(detected_customers & truth_customers)
l3_fp = len(detected_customers - truth_customers)
l3_fn = len(truth_customers   - detected_customers)
l3_precision = l3_tp / len(detected_customers) if detected_customers else 0
l3_recall    = l3_tp / len(truth_customers)    if truth_customers    else 0

print(f"  High-value churned customers detected : {len(churned)}")
print(f"  True pos     : {l3_tp} / {len(truth_customers)} injected")
print(f"  Precision    : {l3_precision:.1%}  |  Recall: {l3_recall:.1%}")
print(f"  Revenue at risk from churned customers: "
      f"R${churned['lifetime_value'].sum():,.0f}")

churned.to_csv("data/outputs/anomalies/detected_L3_churn.csv", index=False)
all_detections.append({
    "detector": "L3_silent_churn", "flags": len(churned),
    "true_positives": l3_tp, "false_positives": l3_fp,
    "false_negatives": l3_fn,
    "precision": round(l3_precision, 3), "recall": round(l3_recall, 3)
})


# ══════════════════════════════════════════════════════════════
# DETECTOR 4: Short Lifecycle (L4)
# Signal: seller-category combos where >25% of orders have
#         delivery-to-purchase gap under 30 days
# ══════════════════════════════════════════════════════════════

print("\nRunning Detector 4: Short Lifecycle...")

lifecycle_sql = """
SELECT
    o.order_id,
    i.seller_id,
    COALESCE(x.product_category_name_english, pr.product_category_name) AS category_en,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    CAST(
        JULIANDAY(o.order_delivered_customer_date) -
        JULIANDAY(o.order_purchase_timestamp)
    AS INTEGER)                                             AS lifecycle_days,
    ROUND(p.payment_value, 2)                               AS payment_value

FROM orders o
JOIN order_items i      ON o.order_id   = i.order_id
JOIN products pr        ON i.product_id = pr.product_id
JOIN order_payments p   ON o.order_id   = p.order_id
LEFT JOIN category_xlat x ON pr.product_category_name = x.product_category_name
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_purchase_timestamp      IS NOT NULL
  AND p.payment_value > 0
"""

lifecycle_df = run_query(lifecycle_sql)
lifecycle_df["is_short"] = lifecycle_df["lifecycle_days"] < 30

# Flag seller-category combos with >25% short lifecycle rate
seller_cat = (lifecycle_df
              .groupby(["seller_id","category_en"])
              .agg(
                  total_orders=("order_id","count"),
                  short_orders=("is_short","sum"),
                  revenue     =("payment_value","sum")
              )
              .reset_index())

seller_cat["short_rate_pct"] = (
    100 * seller_cat["short_orders"] / seller_cat["total_orders"]
).round(1)

# Only flag combos with meaningful volume (at least 5 orders)
flagged_sellers = seller_cat[
    (seller_cat["short_rate_pct"] >= 25) &
    (seller_cat["total_orders"]   >= 5)
].copy()

# Evaluate
l4_truth = ground_truth[ground_truth["leakage_type"] == "L4_short_lifecycle"]
truth_order_ids    = set(l4_truth["order_id"])
detected_order_ids = set(
    lifecycle_df[
        lifecycle_df["seller_id"].isin(flagged_sellers["seller_id"]) &
        (lifecycle_df["lifecycle_days"] < 30)
    ]["order_id"]
)

l4_tp = len(detected_order_ids & truth_order_ids)
l4_fp = len(detected_order_ids - truth_order_ids)
l4_fn = len(truth_order_ids    - detected_order_ids)
l4_precision = l4_tp / len(detected_order_ids) if detected_order_ids else 0
l4_recall    = l4_tp / len(truth_order_ids)    if truth_order_ids    else 0

print(f"  Seller-category combos flagged : {len(flagged_sellers)}")
print(f"  Orders affected                : {len(detected_order_ids)}")
print(f"  True pos     : {l4_tp} / {len(truth_order_ids)} injected")
print(f"  Precision    : {l4_precision:.1%}  |  Recall: {l4_recall:.1%}")

flagged_sellers.to_csv(
    "data/outputs/anomalies/detected_L4_lifecycle.csv", index=False
)
all_detections.append({
    "detector": "L4_short_lifecycle", "flags": len(flagged_sellers),
    "true_positives": l4_tp, "false_positives": l4_fp,
    "false_negatives": l4_fn,
    "precision": round(l4_precision, 3), "recall": round(l4_recall, 3)
})


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL EVALUATION — ALL 4 DETECTORS")
print("="*60)

summary = pd.DataFrame(all_detections)
summary["f1_score"] = (
    2 * summary["precision"] * summary["recall"]
    / (summary["precision"] + summary["recall"]).replace(0, np.nan)
).round(3).fillna(0)

print(summary[["detector","flags","true_positives",
               "precision","recall","f1_score"]].to_string(index=False))

summary.to_csv("data/outputs/anomalies/final_evaluation.csv", index=False)
print("\nSaved → data/outputs/anomalies/final_evaluation.csv")