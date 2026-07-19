# 06_core_metrics.py
# PURPOSE: Calculate the 5 fundamental revenue KPIs.
# These become the "executive summary" numbers in your dashboard.

import os
from db_helper import run_query, save_query

os.makedirs("data/outputs", exist_ok=True)

# ── METRIC 1: Overall GMV and Net Revenue ─────────────────────
# WHY: This is the first number any CFO asks about.
# We separate delivered orders (real revenue) from cancelled ones.

gmv_sql = """
SELECT
    COUNT(DISTINCT o.order_id)                          AS total_orders,
    COUNT(DISTINCT o.customer_id)                       AS total_customers,

    -- GMV = total payment value across ALL non-cancelled orders
    ROUND(SUM(CASE WHEN o.order_status != 'canceled'
                   THEN p.payment_value ELSE 0 END), 2) AS gmv,

    -- Net revenue = payment value for DELIVERED orders only
    ROUND(SUM(CASE WHEN o.order_status = 'delivered'
                   THEN p.payment_value ELSE 0 END), 2) AS net_revenue,

    -- Cancelled revenue = what was lost to cancellations
    ROUND(SUM(CASE WHEN o.order_status = 'canceled'
                   THEN p.payment_value ELSE 0 END), 2) AS cancelled_revenue,

    -- AOV = average order value (delivered orders only)
    ROUND(AVG(CASE WHEN o.order_status = 'delivered'
                   THEN p.payment_value END), 2)        AS avg_order_value,

    -- Cancellation rate
    ROUND(100.0 * SUM(CASE WHEN o.order_status = 'canceled'
                            THEN 1 ELSE 0 END)
               / COUNT(*), 2)                           AS cancellation_rate_pct

FROM orders o
JOIN order_payments p ON o.order_id = p.order_id
WHERE p.payment_value > 0   -- exclude the 9 zero-value rows we found
"""

gmv = run_query(gmv_sql)
print("=== OVERALL REVENUE KPIs ===")
print(gmv.to_string(index=False))
gmv.to_csv("data/outputs/metric_01_overall_gmv.csv", index=False)


# ── METRIC 2: Monthly Revenue Trend ───────────────────────────
# WHY: Every business tracks MoM (month-over-month) revenue.
# This shows whether the business is growing or declining.

monthly_sql = """
SELECT
    STRFTIME('%Y-%m', o.order_purchase_timestamp)       AS year_month,
    COUNT(DISTINCT o.order_id)                          AS orders_count,
    ROUND(SUM(p.payment_value), 2)                      AS monthly_revenue,
    ROUND(AVG(p.payment_value), 2)                      AS monthly_aov,
    COUNT(DISTINCT o.customer_id)                       AS unique_customers

FROM orders o
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
  AND p.payment_value > 0
  AND o.order_purchase_timestamp IS NOT NULL
GROUP BY year_month
ORDER BY year_month
"""

monthly = run_query(monthly_sql)
print("\n=== MONTHLY REVENUE TREND ===")
print(monthly.to_string(index=False))
monthly.to_csv("data/outputs/metric_02_monthly_revenue.csv", index=False)


# ── METRIC 3: Refund Rate by Month ────────────────────────────
# WHY: This is Leakage L2 foundation. A rising refund rate
# is the first warning sign that something is wrong.

refund_sql = """
SELECT
    STRFTIME('%Y-%m', o.order_purchase_timestamp)       AS year_month,
    COUNT(*)                                            AS total_orders,

    SUM(CASE WHEN o.order_status = 'canceled'
             THEN 1 ELSE 0 END)                         AS cancelled_orders,

    ROUND(100.0 * SUM(CASE WHEN o.order_status = 'canceled'
                            THEN 1 ELSE 0 END)
               / COUNT(*), 2)                           AS refund_rate_pct

FROM orders o
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY year_month
ORDER BY year_month
"""

refund = run_query(refund_sql)
print("\n=== MONTHLY REFUND RATE ===")
print(refund.to_string(index=False))
refund.to_csv("data/outputs/metric_03_refund_rate.csv", index=False)


# ── METRIC 4: Discount Depth ──────────────────────────────────
# WHY: This is Leakage L1 foundation. Discount depth measures
# how much the actual payment differs from the item list price.
# High discount depth = money left on the table.

discount_sql = """
SELECT
    STRFTIME('%Y-%m', o.order_purchase_timestamp)       AS year_month,
    COUNT(DISTINCT o.order_id)                          AS orders_count,

    ROUND(SUM(i.price + i.freight_value), 2)            AS total_list_price,
    ROUND(SUM(p.payment_value), 2)                      AS total_paid,

    -- Discount depth = how much less was paid vs list price
    ROUND(100.0 * (SUM(i.price + i.freight_value) - SUM(p.payment_value))
               /  SUM(i.price + i.freight_value), 2)   AS discount_depth_pct

FROM orders o
JOIN order_items i    ON o.order_id = i.order_id
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
  AND p.payment_value > 0
GROUP BY year_month
ORDER BY year_month
"""

discount = run_query(discount_sql)
print("\n=== DISCOUNT DEPTH BY MONTH ===")
print(discount.to_string(index=False))
discount.to_csv("data/outputs/metric_04_discount_depth.csv", index=False)


# ── METRIC 5: Revenue by Product Category ─────────────────────
# WHY: This is your first dimension slice. Leakage always hides
# in specific segments — not in the overall number.

category_sql = """
SELECT
    COALESCE(x.product_category_name_english, p.product_category_name, 'unknown') AS category_en,
    COUNT(DISTINCT o.order_id)                                      AS orders_count,
    ROUND(SUM(pay.payment_value), 2)                               AS total_revenue,
    ROUND(AVG(pay.payment_value), 2)                               AS avg_order_value,
    ROUND(100.0 * SUM(CASE WHEN o.order_status = 'canceled'
                            THEN 1 ELSE 0 END)
               / COUNT(*), 2)                                      AS cancel_rate_pct

FROM orders o
JOIN order_items i     ON o.order_id   = i.order_id
JOIN products p        ON i.product_id = p.product_id
JOIN order_payments pay ON o.order_id  = pay.order_id
LEFT JOIN category_xlat x
       ON p.product_category_name = x.product_category_name
WHERE pay.payment_value > 0
GROUP BY category_en
ORDER BY total_revenue DESC
LIMIT 20
"""

category = run_query(category_sql)
print("\n=== TOP 20 CATEGORIES BY REVENUE ===")
print(category.to_string(index=False))
category.to_csv("data/outputs/metric_05_category_revenue.csv", index=False)

print("\nAll 5 core metrics saved to data/outputs/")