# 04_data_quality_report.py
# PURPOSE: Document every data quality issue found.
# This becomes a section in your GitHub README and a talking point in interviews.

import pandas as pd
import numpy as np
import os
os.makedirs("reports", exist_ok=True)

orders   = pd.read_csv("data/raw/olist_orders_dataset.csv",
                        parse_dates=["order_purchase_timestamp",
                                     "order_approved_at",
                                     "order_delivered_carrier_date",
                                     "order_delivered_customer_date",
                                     "order_estimated_delivery_date"])
items    = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")

report = []

# ── Check 1: Null rates per column ────────────────────────────
for col in orders.columns:
    null_pct = orders[col].isnull().mean() * 100
    if null_pct > 0:
        report.append({
            "table": "orders", "column": col,
            "issue": "nulls", "pct_affected": round(null_pct, 2),
            "note": f"{null_pct:.1f}% of rows are null"
        })

# ── Check 2: Orders with status=delivered but no delivery date ─
delivered_no_date = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_delivered_customer_date"].isnull())
]
report.append({
    "table": "orders", "column": "order_delivered_customer_date",
    "issue": "logical inconsistency",
    "pct_affected": round(len(delivered_no_date)/len(orders)*100, 2),
    "note": f"{len(delivered_no_date)} orders marked delivered but have no delivery date"
})

# ── Check 3: Delivery before purchase (impossible dates) ──────
orders["delivery_before_purchase"] = (
    orders["order_delivered_customer_date"] < orders["order_purchase_timestamp"]
)
impossible = orders["delivery_before_purchase"].sum()
report.append({
    "table": "orders", "column": "order_delivered_customer_date",
    "issue": "impossible date",
    "pct_affected": round(impossible/len(orders)*100, 4),
    "note": f"{impossible} orders delivered before they were purchased"
})

# ── Check 4: Payments where value = 0 ─────────────────────────
zero_payments = payments[payments["payment_value"] == 0]
report.append({
    "table": "order_payments", "column": "payment_value",
    "issue": "zero value",
    "pct_affected": round(len(zero_payments)/len(payments)*100, 2),
    "note": f"{len(zero_payments)} payment rows with value = 0"
})

# ── Check 5: Items with price = 0 or negative ─────────────────
zero_price = items[items["price"] <= 0]
report.append({
    "table": "order_items", "column": "price",
    "issue": "zero or negative price",
    "pct_affected": round(len(zero_price)/len(items)*100, 2),
    "note": f"{len(zero_price)} items with price <= 0"
})

# ── Check 6: Duplicate order_ids in orders table ──────────────
dup_orders = orders["order_id"].duplicated().sum()
report.append({
    "table": "orders", "column": "order_id",
    "issue": "duplicates",
    "pct_affected": round(dup_orders/len(orders)*100, 4),
    "note": f"{dup_orders} duplicate order_ids found"
})

# ── Print + save ──────────────────────────────────────────────
report_df = pd.DataFrame(report)
print("\n===== DATA QUALITY REPORT =====")
print(report_df.to_string(index=False))
report_df.to_csv("reports/data_quality_report.csv", index=False)
print("\nSaved to reports/data_quality_report.csv")