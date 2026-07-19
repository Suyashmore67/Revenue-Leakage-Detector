# 07_segment_trends.py
# PURPOSE: Revenue trending by segment using SQL window functions.
# This is the analytical layer that feeds into anomaly detection.

from db_helper import run_query, save_query
import os
os.makedirs("data/outputs", exist_ok=True)


# ── TREND 1: Weekly Revenue with MoM comparison ───────────────
# LAG(1) = previous week's revenue
# This lets you see week-over-week change in SQL itself

weekly_trend_sql = """
WITH weekly_revenue AS (
    SELECT
        STRFTIME('%Y-W%W', o.order_purchase_timestamp)  AS year_week,
        STRFTIME('%Y', o.order_purchase_timestamp)      AS year,
        CAST(STRFTIME('%W', o.order_purchase_timestamp)
             AS INTEGER)                                AS week_num,
        ROUND(SUM(p.payment_value), 2)                  AS weekly_revenue,
        COUNT(DISTINCT o.order_id)                      AS orders_count

    FROM orders o
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
      AND p.payment_value > 0
      AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY year_week
)

SELECT
    year_week,
    weekly_revenue,
    orders_count,

    -- LAG: previous week's revenue (window function)
    LAG(weekly_revenue, 1) OVER (ORDER BY year_week)    AS prev_week_revenue,

    -- Week-over-week change
    ROUND(weekly_revenue -
          LAG(weekly_revenue, 1) OVER (ORDER BY year_week), 2) AS wow_change,

    -- WoW % change
    ROUND(100.0 * (weekly_revenue -
                   LAG(weekly_revenue, 1) OVER (ORDER BY year_week))
               /  LAG(weekly_revenue, 1) OVER (ORDER BY year_week), 2) AS wow_pct

FROM weekly_revenue
ORDER BY year_week
"""

weekly = run_query(weekly_trend_sql)
print("=== WEEKLY REVENUE WITH WoW CHANGE ===")
print(weekly.tail(10).to_string(index=False))
weekly.to_csv("data/outputs/trend_01_weekly_revenue.csv", index=False)


# ── TREND 2: Revenue by Category × Week ───────────────────────
# WHY: Overall revenue can look fine while one category bleeds.
# This query lets you see each category's weekly revenue separately.
# This is the direct input to your anomaly detector in Phase 3.

category_weekly_sql = """
SELECT
    STRFTIME('%Y-W%W', o.order_purchase_timestamp)                  AS year_week,
    COALESCE(x.product_category_name_english, p.product_category_name, 'unknown')   AS category_en,
    COUNT(DISTINCT o.order_id)                                       AS orders_count,
    ROUND(SUM(pay.payment_value), 2)                                AS weekly_revenue,
    ROUND(AVG(pay.payment_value), 2)                                AS avg_order_value,
    SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END)   AS cancellations

FROM orders o
JOIN order_items i      ON o.order_id   = i.order_id
JOIN products p         ON i.product_id = p.product_id
JOIN order_payments pay ON o.order_id   = pay.order_id
LEFT JOIN category_xlat x ON p.product_category_name = x.product_category_name
WHERE pay.payment_value > 0
  AND o.order_purchase_timestamp IS NOT NULL
GROUP BY year_week, category_en
ORDER BY year_week, weekly_revenue DESC
"""

cat_weekly = run_query(category_weekly_sql)
print("\n=== CATEGORY × WEEK REVENUE (sample) ===")
print(cat_weekly.head(15).to_string(index=False))
cat_weekly.to_csv("data/outputs/trend_02_category_weekly.csv", index=False)


# ── TREND 3: Revenue by Seller State × Week ───────────────────
# WHY: Geographic leakage — some regions suddenly drop off.
# Sellers in a specific state returning bad products, for example.

geo_weekly_sql = """
SELECT
    STRFTIME('%Y-W%W', o.order_purchase_timestamp)  AS year_week,
    s.seller_state,
    COUNT(DISTINCT o.order_id)                      AS orders_count,
    ROUND(SUM(pay.payment_value), 2)               AS weekly_revenue,
    SUM(CASE WHEN o.order_status = 'canceled'
             THEN 1 ELSE 0 END)                    AS cancellations

FROM orders o
JOIN order_items i      ON o.order_id   = i.order_id
JOIN sellers s          ON i.seller_id  = s.seller_id
JOIN order_payments pay ON o.order_id   = pay.order_id
WHERE pay.payment_value > 0
  AND o.order_purchase_timestamp IS NOT NULL
GROUP BY year_week, s.seller_state
ORDER BY year_week, weekly_revenue DESC
"""

geo_weekly = run_query(geo_weekly_sql)
print("\n=== GEO × WEEK REVENUE (sample) ===")
print(geo_weekly.head(10).to_string(index=False))
geo_weekly.to_csv("data/outputs/trend_03_geo_weekly.csv", index=False)


# ── TREND 4: Customer Tier Revenue ────────────────────────────
# WHY: This is Leakage L3 foundation. We split customers into
# High / Mid / Low value tiers and track revenue from each tier.
# Losing high-value tier revenue is the most dangerous leakage.

customer_tier_sql = """
WITH customer_ltv AS (
    SELECT
        c.customer_unique_id,
        ROUND(SUM(p.payment_value), 2) AS lifetime_value
    FROM customers c
    JOIN orders o     ON c.customer_id = o.customer_id
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
      AND p.payment_value > 0
    GROUP BY c.customer_unique_id
),
customer_tiers AS (
    SELECT
        customer_unique_id,
        lifetime_value,
        CASE
            WHEN lifetime_value >= (SELECT PERCENTILE_VALUE FROM (
                SELECT lifetime_value,
                       PERCENT_RANK() OVER (ORDER BY lifetime_value) AS pct
                FROM customer_ltv
            ) WHERE pct >= 0.80 LIMIT 1)
            THEN 'High'
            WHEN lifetime_value >= (SELECT PERCENTILE_VALUE FROM (
                SELECT lifetime_value,
                       PERCENT_RANK() OVER (ORDER BY lifetime_value) AS pct
                FROM customer_ltv
            ) WHERE pct >= 0.40 LIMIT 1)
            THEN 'Mid'
            ELSE 'Low'
        END AS tier
    FROM customer_ltv
)
SELECT tier,
       COUNT(*)                         AS customer_count,
       ROUND(SUM(lifetime_value), 2)    AS total_revenue,
       ROUND(AVG(lifetime_value), 2)    AS avg_ltv
FROM customer_tiers
GROUP BY tier
ORDER BY avg_ltv DESC
"""

# NOTE: SQLite doesn't support PERCENTILE directly.
# Use Python for customer tiering instead:
import pandas as pd

ltv_sql = """
SELECT
    c.customer_unique_id,
    ROUND(SUM(p.payment_value), 2) AS lifetime_value
FROM customers c
JOIN orders o        ON c.customer_id  = o.customer_id
JOIN order_payments p ON o.order_id   = p.order_id
WHERE o.order_status  = 'delivered'
  AND p.payment_value > 0
GROUP BY c.customer_unique_id
"""

ltv_df = run_query(ltv_sql)
ltv_df["tier"] = pd.cut(
    ltv_df["lifetime_value"],
    bins=[0,
          ltv_df["lifetime_value"].quantile(0.40),
          ltv_df["lifetime_value"].quantile(0.80),
          float("inf")],
    labels=["Low", "Mid", "High"]
)

tier_summary = (
    ltv_df.groupby("tier", observed=True)
    .agg(
        customer_count=("customer_unique_id", "count"),
        total_revenue=("lifetime_value", "sum"),
        avg_ltv=("lifetime_value", "mean")
    )
    .round(2)
    .reset_index()
    .sort_values("avg_ltv", ascending=False)
)

print("\n=== CUSTOMER TIER SUMMARY ===")
print(tier_summary.to_string(index=False))
ltv_df.to_csv("data/outputs/trend_04_customer_ltv.csv", index=False)
tier_summary.to_csv("data/outputs/trend_04_customer_tiers.csv", index=False)

print("\nAll 4 segment trends saved to data/outputs/")