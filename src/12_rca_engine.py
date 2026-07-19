# 12_rca_engine.py
# PURPOSE: When an anomaly is flagged, automatically drill down
# to find the root cause dimension. This is the key differentiator
# of the project — no dashboard does this automatically.

import pandas as pd
import numpy as np
import sqlite3
import os

os.makedirs("data/outputs/rca", exist_ok=True)

DB_PATH = "data/db/olist.db"

def run_query(sql):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)

# Load high-confidence anomalies — these are our RCA targets
anomalies = pd.read_csv("data/outputs/anomalies/high_confidence_anomalies.csv")

# ── RCA Function ───────────────────────────────────────────────
# For each flagged week + category, we drill down across 3 dimensions:
# Dimension 1: Which seller state is underperforming?
# Dimension 2: Which specific seller is underperforming?
# Dimension 3: Is a specific customer tier contributing to the drop?

def run_rca(year_week: str, category: str) -> dict:
    """
    Given a flagged week and category, find the root cause
    by slicing across seller_state, seller_id, and customer tier.
    Returns a dict with top contributing dimension for each level.
    """

    # ── Drill 1: By seller state ──────────────────────────────
    state_sql = f"""
    SELECT
        s.seller_state,
        COUNT(DISTINCT o.order_id)          AS orders,
        ROUND(SUM(p.payment_value), 2)      AS revenue,
        SUM(CASE WHEN o.order_status = 'canceled'
                 THEN 1 ELSE 0 END)         AS cancellations

    FROM orders o
    JOIN order_items i      ON o.order_id   = i.order_id
    JOIN products pr        ON i.product_id = pr.product_id
    JOIN sellers s          ON i.seller_id  = s.seller_id
    JOIN order_payments p   ON o.order_id   = p.order_id
    LEFT JOIN category_xlat x
           ON pr.product_category_name = x.product_category_name

    WHERE STRFTIME('%Y-W%W', o.order_purchase_timestamp) = '{year_week}'
      AND COALESCE(x.product_category_name_english,
                   pr.product_category_name) = '{category}'
      AND p.payment_value > 0
    GROUP BY s.seller_state
    ORDER BY revenue DESC
    """
    state_df = run_query(state_sql)

    # ── Drill 2: By seller ────────────────────────────────────
    seller_sql = f"""
    SELECT
        s.seller_id,
        s.seller_state,
        s.seller_city,
        COUNT(DISTINCT o.order_id)          AS orders,
        ROUND(SUM(p.payment_value), 2)      AS revenue,
        SUM(CASE WHEN o.order_status = 'canceled'
                 THEN 1 ELSE 0 END)         AS cancellations,
        ROUND(100.0 * SUM(CASE WHEN o.order_status = 'canceled'
                               THEN 1 ELSE 0 END)
                   / COUNT(*), 1)           AS cancel_rate_pct

    FROM orders o
    JOIN order_items i      ON o.order_id   = i.order_id
    JOIN products pr        ON i.product_id = pr.product_id
    JOIN sellers s          ON i.seller_id  = s.seller_id
    JOIN order_payments p   ON o.order_id   = p.order_id
    LEFT JOIN category_xlat x
           ON pr.product_category_name = x.product_category_name

    WHERE STRFTIME('%Y-W%W', o.order_purchase_timestamp) = '{year_week}'
      AND COALESCE(x.product_category_name_english,
                   pr.product_category_name) = '{category}'
      AND p.payment_value > 0
    GROUP BY s.seller_id
    ORDER BY cancellations DESC, revenue ASC
    LIMIT 5
    """
    seller_df = run_query(seller_sql)

    # ── Drill 3: By customer tier ─────────────────────────────
    # Load LTV tiers from Phase 2
    ltv_df = pd.read_csv("data/outputs/trend_04_customer_ltv.csv")
    ltv_df["tier"] = pd.cut(
        ltv_df["lifetime_value"],
        bins=[0,
              ltv_df["lifetime_value"].quantile(0.40),
              ltv_df["lifetime_value"].quantile(0.80),
              float("inf")],
        labels=["Low","Mid","High"]
    )
    tier_map = ltv_df.set_index("customer_unique_id")["tier"].to_dict()

    tier_sql = f"""
    SELECT
        c.customer_unique_id,
        o.order_id,
        p.payment_value

    FROM orders o
    JOIN customers c        ON o.customer_id = c.customer_id
    JOIN order_payments p   ON o.order_id    = p.order_id
    JOIN order_items i      ON o.order_id    = i.order_id
    JOIN products pr        ON i.product_id  = pr.product_id
    LEFT JOIN category_xlat x
           ON pr.product_category_name = x.product_category_name

    WHERE STRFTIME('%Y-W%W', o.order_purchase_timestamp) = '{year_week}'
      AND COALESCE(x.product_category_name_english,
                   pr.product_category_name) = '{category}'
      AND p.payment_value > 0
    """
    tier_orders = run_query(tier_sql)
    tier_orders["tier"] = tier_orders["customer_unique_id"].map(tier_map)
    tier_summary = (tier_orders.groupby("tier", observed=True)["payment_value"]
                    .agg(["count","sum"])
                    .rename(columns={"count":"orders","sum":"revenue"})
                    .round(2))

    return {
        "by_state":  state_df,
        "by_seller": seller_df,
        "by_tier":   tier_summary
    }


# ── Run RCA on top 10 anomalies ───────────────────────────────
top_anomalies = (anomalies
                 .sort_values("revenue_at_risk", ascending=False)
                 .head(10))

all_rca_results = []

print("=== ROOT CAUSE ANALYSIS ===\n")

for _, row in top_anomalies.iterrows():
    week     = row["year_week"]
    category = row["category_en"]
    gap      = row.get("revenue_gap", row.get("revenue_at_risk", 0))

    print(f"{'─'*60}")
    print(f"ANOMALY: {category} | Week: {week}")
    print(f"Revenue gap: R${gap:,.0f} | Z-score: {row['z_score']:.2f}")

    rca = run_rca(week, category)

    # Top contributing state
    if not rca["by_state"].empty:
        top_state = rca["by_state"].iloc[0]
        print(f"\n  Top state   : {top_state['seller_state']}"
              f" (R${top_state['revenue']:,.0f} revenue,"
              f" {top_state['cancellations']} cancellations)")

    # Top problem seller
    if not rca["by_seller"].empty:
        top_seller = rca["by_seller"].iloc[0]
        print(f"  Top seller  : {top_seller['seller_id'][:12]}..."
              f" in {top_seller['seller_city']}"
              f" — cancel rate {top_seller['cancel_rate_pct']}%")

    # Customer tier breakdown
    if not rca["by_tier"].empty:
        print(f"  Tier impact :")
        for tier, tier_row in rca["by_tier"].iterrows():
            print(f"    {tier:4s} tier → "
                  f"{int(tier_row['orders'])} orders, "
                  f"R${tier_row['revenue']:,.0f}")

    # Build summary row for saving
    top_state_name  = (rca["by_state"].iloc[0]["seller_state"]
                        if not rca["by_state"].empty else "unknown")
    top_seller_id   = (rca["by_seller"].iloc[0]["seller_id"]
                        if not rca["by_seller"].empty else "unknown")
    top_cancel_rate = (rca["by_seller"].iloc[0]["cancel_rate_pct"]
                        if not rca["by_seller"].empty else 0)

    all_rca_results.append({
        "year_week":         week,
        "category":          category,
        "revenue_gap":       round(gap, 2),
        "z_score":           round(row["z_score"], 2),
        "root_cause_state":  top_state_name,
        "root_cause_seller": top_seller_id,
        "seller_cancel_rate":top_cancel_rate,
        "recommended_action": (
            f"Investigate seller {top_seller_id[:8]} in {top_state_name} — "
            f"cancel rate {top_cancel_rate}% in {category} during {week}"
        )
    })

# ── Save RCA report ───────────────────────────────────────────
rca_report = pd.DataFrame(all_rca_results)
rca_report.to_csv("data/outputs/rca/rca_report.csv", index=False)

print(f"\n{'='*60}")
print(f"RCA complete for {len(rca_report)} anomalies")
print("Saved → data/outputs/rca/rca_report.csv")