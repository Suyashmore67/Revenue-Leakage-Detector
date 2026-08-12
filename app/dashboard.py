# app/dashboard.py
# PURPOSE: Main Streamlit entry point.
# Run with: streamlit run app/dashboard.py

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_data, get_summary_kpis
from components  import exec_view, analyst_view

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Revenue Leakage Detector",
    page_icon  = "📉",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load data ─────────────────────────────────────────────────
data = load_all_data()
kpis = get_summary_kpis(data)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Revenue Leakage Detector")
    st.caption("Olist E-Commerce · 2016–2018")
    st.divider()

    page = st.radio(
        "View",
        options=["Executive Summary", "Analyst Drill-Down"],
        index=0
    )

    st.divider()
    st.markdown("**Dataset stats**")
    st.caption(f"Monthly records: {len(data['monthly'])}")
    st.caption(f"Categories tracked: "
               f"{data['baseline']['category_en'].nunique()}")
    st.caption(f"Anomaly flags: "
               f"{data['zscore']['is_anomaly'].sum() if 'is_anomaly' in data['zscore'].columns else len(data['zscore'])}")

    st.divider()
    st.markdown(
        "Built by Suyash More."
        "[GitHub](https://github.com/Suyashmore67/revenue-leakage-detector)"
        "(revenue-leakage-detector)"
    )

# ── Render selected page ──────────────────────────────────────
if page == "Executive Summary":
    exec_view.render(data, kpis)
else:
    analyst_view.render(data)