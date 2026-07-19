# app/data_loader.py
# PURPOSE: Single place to load all data files.
# Uses Streamlit caching so data is only read once,
# not on every user interaction.

import pandas as pd
import streamlit as st
import os

DATA_BASE = os.path.join(os.path.dirname(__file__), "..", "data")

@st.cache_data
def load_all_data():
    """Load and return all data needed by the dashboard."""

    # ── Anomaly data ──────────────────────────────────────────
    baseline = pd.read_csv(
        f"{DATA_BASE}/outputs/baseline_category_weekly.csv"
    )

    zscore = pd.read_csv(
        f"{DATA_BASE}/outputs/anomalies/zscore_anomalies.csv"
    )

    l1 = pd.read_csv(
        f"{DATA_BASE}/outputs/anomalies/detected_L1_fixed.csv"
    )
    l2 = pd.read_csv(
        f"{DATA_BASE}/outputs/anomalies/detected_L2_refund.csv"
    )
    l3 = pd.read_csv(
        f"{DATA_BASE}/outputs/anomalies/detected_L3_fixed.csv"
    )
    l4 = pd.read_csv(
        f"{DATA_BASE}/outputs/anomalies/detected_L4_fixed.csv"
    )

    rca = pd.read_csv(
        f"{DATA_BASE}/outputs/rca/rca_report.csv"
    )

    # ── Revenue metrics ───────────────────────────────────────
    monthly    = pd.read_csv(f"{DATA_BASE}/outputs/metric_02_monthly_revenue.csv")
    category   = pd.read_csv(f"{DATA_BASE}/outputs/metric_05_category_revenue.csv")
    cat_weekly = pd.read_csv(f"{DATA_BASE}/outputs/trend_02_category_weekly.csv")
    tiers      = pd.read_csv(f"{DATA_BASE}/outputs/trend_04_customer_tiers.csv")
    eval_df    = pd.read_csv(
        f"{DATA_BASE}/outputs/anomalies/final_evaluation_fixed.csv"
    )

    # ── Parse dates ───────────────────────────────────────────
    monthly["year_month"] = pd.to_datetime(
        monthly["year_month"], format="%Y-%m"
    )

    return {
        "baseline":   baseline,
        "zscore":     zscore,
        "l1":         l1,
        "l2":         l2,
        "l3":         l3,
        "l4":         l4,
        "rca":        rca,
        "monthly":    monthly,
        "category":   category,
        "cat_weekly": cat_weekly,
        "tiers":      tiers,
        "eval_df":    eval_df,
    }


def get_summary_kpis(data: dict) -> dict:
    """Compute the 4 headline KPI numbers for the executive view."""

    l1_risk = data["l1"]["actual_paid"].sum()
    l3_risk = data["l3"]["lifetime_value"].sum()
    l4_risk = data["l4"]["payment_value"].sum()
    total_risk = l1_risk + l3_risk + l4_risk

    active_anomalies = (
        len(data["l1"]) +
        len(data["l2"]) +
        len(data["l3"]) +
        len(data["l4"])
    )

    top_category = (
        data["zscore"]
        .groupby("category_en")["revenue_gap"]
        .sum()
        .idxmax()
        if len(data["zscore"]) > 0 else "N/A"
    )

    worst_week = (
        data["zscore"]
        .groupby("year_week")["revenue_gap"]
        .sum()
        .idxmax()
        if len(data["zscore"]) > 0 else "N/A"
    )

    return {
        "total_risk":       total_risk,
        "active_anomalies": active_anomalies,
        "top_category":     top_category,
        "worst_week":       worst_week,
        "l1_risk":          l1_risk,
        "l3_risk":          l3_risk,
        "l4_risk":          l4_risk,
    }