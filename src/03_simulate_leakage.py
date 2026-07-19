# 03_simulate_leakage.py
# PURPOSE: Inject realistic leakage events into clean data.
# Output: leakage_injected_orders.csv + leakage_ground_truth.csv (the labels)

import pandas as pd
import numpy as np
import random
import os
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)

random.seed(42)
np.random.seed(42)

orders    = pd.read_csv("data/raw/olist_orders_dataset.csv")
items     = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments  = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
products  = pd.read_csv("data/raw/olist_products_dataset.csv")

# Convert dates
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

ground_truth = []  # We'll store every injected anomaly here


# ── LEAKAGE 1: Discount abuse ─────────────────────────────────
# Pick 200 random orders, slash their payment value by 40-60%
discount_abuse_ids = payments.sample(200, random_state=42)["order_id"].tolist()

payments_modified = payments.copy()
mask = payments_modified["order_id"].isin(discount_abuse_ids)
payments_modified.loc[mask, "payment_value"] *= np.random.uniform(0.4, 0.6,
                                                    size=mask.sum())

for oid in discount_abuse_ids:
    ground_truth.append({
        "order_id": oid, "leakage_type": "L1_discount_abuse",
        "injected": True
    })

print(f"L1 Discount abuse: injected into {len(discount_abuse_ids)} orders")


# ── LEAKAGE 2: Refund spike in a category ─────────────────────
# Pick one category and cancel 60% of its orders in weeks 30-35
items_with_cat = items.merge(products[["product_id","product_category_name"]],
                              on="product_id")
target_category = "cama_mesa_banho"   # bed/bath — large category
cat_orders = items_with_cat[
    items_with_cat["product_category_name"] == target_category
]["order_id"].unique()

# Get orders in "spike weeks" (filter by week of year 30–35)
orders["week"] = orders["order_purchase_timestamp"].dt.isocalendar().week
spike_weeks = orders[orders["week"].between(30, 35)]["order_id"].tolist()
spike_targets = list(set(cat_orders) & set(spike_weeks))
spike_sample = random.sample(spike_targets, min(150, len(spike_targets)))

orders_modified = orders.copy()
orders_modified.loc[
    orders_modified["order_id"].isin(spike_sample), "order_status"
] = "canceled"

for oid in spike_sample:
    ground_truth.append({
        "order_id": oid, "leakage_type": "L2_refund_spike",
        "injected": True
    })

print(f"L2 Refund spike: injected {len(spike_sample)} cancellations in '{target_category}'")


# ── LEAKAGE 3: Silent churn of high-value customers ──────────
# Identify top 20% customers by spend, then erase their recent orders
payment_totals = payments.groupby("order_id")["payment_value"].sum().reset_index()
order_customer = orders[["order_id","customer_id","order_purchase_timestamp"]].copy()
customer_ltv = (
    order_customer
    .merge(payment_totals, on="order_id")
    .groupby("customer_id")["payment_value"].sum()
    .reset_index()
    .rename(columns={"payment_value": "ltv"})
)
top20_threshold = customer_ltv["ltv"].quantile(0.80)
top20_customers = customer_ltv[customer_ltv["ltv"] >= top20_threshold]["customer_id"]

# Simulate churn: delete orders placed in last 6 months for these customers
cutoff = orders["order_purchase_timestamp"].max() - pd.DateOffset(months=6)
churn_order_ids = orders[
    (orders["customer_id"].isin(top20_customers)) &
    (orders["order_purchase_timestamp"] >= cutoff)
]["order_id"].tolist()

churn_sample = random.sample(churn_order_ids, min(300, len(churn_order_ids)))
orders_modified = orders_modified[~orders_modified["order_id"].isin(churn_sample)]

for oid in churn_sample:
    ground_truth.append({
        "order_id": oid, "leakage_type": "L3_silent_churn",
        "injected": True
    })

print(f"L3 Silent churn: removed {len(churn_sample)} recent orders from top-20% customers")


# ── LEAKAGE 4: Short lifecycle seller-category combos ─────────
# Pick one seller, set all their delivered orders to return within 20 days
sellers_in_items = items["seller_id"].value_counts()
problem_seller = sellers_in_items.index[5]  # pick a mid-volume seller

problem_orders = items[items["seller_id"] == problem_seller]["order_id"].unique()
short_life_sample = random.sample(list(problem_orders), min(100, len(problem_orders)))

orders_modified.loc[
    orders_modified["order_id"].isin(short_life_sample),
    "order_delivered_customer_date"
] = (
    orders_modified.loc[
        orders_modified["order_id"].isin(short_life_sample),
        "order_purchase_timestamp"
    ] + pd.Timedelta(days=15)
).values

for oid in short_life_sample:
    ground_truth.append({
        "order_id": oid, "leakage_type": "L4_short_lifecycle",
        "injected": True
    })

print(f"L4 Short lifecycle: injected 15-day delivery for {len(short_life_sample)} orders"
      f" from seller {problem_seller}")


# ── Save everything ───────────────────────────────────────────
orders_modified.to_csv("data/processed/orders_with_leakage.csv", index=False)
payments_modified.to_csv("data/processed/payments_with_leakage.csv", index=False)

ground_truth_df = pd.DataFrame(ground_truth)
ground_truth_df.to_csv("data/processed/leakage_ground_truth.csv", index=False)

print(f"\nTotal injected leakage events: {len(ground_truth_df)}")
print(ground_truth_df["leakage_type"].value_counts())