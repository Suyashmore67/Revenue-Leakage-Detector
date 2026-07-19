# app/components/exec_view.py
# PURPOSE: Executive summary page — clean, simple, no jargon.
# Audience: CFO, VP, non-technical stakeholder.

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render_kpi_cards(kpis: dict):
    """4 headline KPI cards across the top."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label    = "Total Revenue at Risk",
            value    = f"R${kpis['total_risk']:,.0f}",
            delta    = "Across 4 leakage types",
            delta_color = "off"
        )
    with c2:
        st.metric(
            label = "Active Anomaly Flags",
            value = f"{kpis['active_anomalies']:,}",
            delta = "Orders + customers affected",
            delta_color = "off"
        )
    with c3:
        st.metric(
            label = "Most Affected Category",
            value = kpis["top_category"][:20],
            delta = "By revenue gap",
            delta_color = "off"
        )
    with c4:
        st.metric(
            label = "Worst Revenue Week",
            value = kpis["worst_week"],
            delta = "Highest single-week gap",
            delta_color = "off"
        )


def render_revenue_trend(monthly: pd.DataFrame):
    """Monthly revenue trend line with anomaly overlay."""
    st.subheader("Monthly Revenue Trend")

    fig = go.Figure()

    # Revenue line
    fig.add_trace(go.Scatter(
        x    = monthly["year_month"],
        y    = monthly["monthly_revenue"],
        mode = "lines+markers",
        name = "Monthly Revenue",
        line = dict(color="#7F77DD", width=2.5),
        marker = dict(size=5)
    ))

    # Rolling average
    monthly["rolling_avg"] = (monthly["monthly_revenue"]
                               .rolling(3, min_periods=1).mean())
    fig.add_trace(go.Scatter(
        x    = monthly["year_month"],
        y    = monthly["rolling_avg"],
        mode = "lines",
        name = "3-month avg",
        line = dict(color="#1D9E75", width=1.5, dash="dash")
    ))

    fig.update_layout(
        height          = 320,
        margin          = dict(l=0, r=0, t=10, b=0),
        legend          = dict(orientation="h", y=1.1),
        plot_bgcolor    = "rgba(0,0,0,0)",
        paper_bgcolor   = "rgba(0,0,0,0)",
        xaxis           = dict(showgrid=False),
        yaxis           = dict(gridcolor="#f0f0f0",
                               tickprefix="R$",
                               tickformat=",.0f"),
        hovermode       = "x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_leakage_breakdown(kpis: dict):
    """Horizontal bar chart — leakage by type."""
    st.subheader("Revenue at Risk by Leakage Type")

    leakage_data = pd.DataFrame({
        "Leakage Type": [
            "Discount Abuse (L1)",
            "Silent Churn (L3)",
            "Short Lifecycle (L4)"
        ],
        "Revenue at Risk": [
            kpis["l1_risk"],
            kpis["l3_risk"],
            kpis["l4_risk"]
        ],
        "Color": ["#7F77DD", "#1D9E75", "#D85A30"]
    }).sort_values("Revenue at Risk", ascending=True)

    fig = go.Figure(go.Bar(
        x           = leakage_data["Revenue at Risk"],
        y           = leakage_data["Leakage Type"],
        orientation = "h",
        marker_color= leakage_data["Color"],
        text        = leakage_data["Revenue at Risk"].apply(
                          lambda x: f"R${x:,.0f}"
                      ),
        textposition= "outside"
    ))

    fig.update_layout(
        height       = 220,
        margin       = dict(l=0, r=80, t=10, b=0),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor= "rgba(0,0,0,0)",
        xaxis        = dict(showgrid=False, showticklabels=False),
        yaxis        = dict(showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_categories(category: pd.DataFrame):
    """Table of top 5 categories by revenue + cancel rate."""
    st.subheader("Top Categories at Risk")

    display = (category
               .sort_values("cancel_rate_pct", ascending=False)
               .head(5)[["category_en", "total_revenue",
                          "orders_count", "cancel_rate_pct"]]
               .rename(columns={
                   "category_en":     "Category",
                   "total_revenue":   "Revenue (R$)",
                   "orders_count":    "Orders",
                   "cancel_rate_pct": "Cancel Rate %"
               })
               .reset_index(drop=True))

    display["Revenue (R$)"] = display["Revenue (R$)"].apply(
        lambda x: f"R${x:,.0f}"
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


def render(data: dict, kpis: dict):
    """Render the full executive view."""
    st.markdown("### Revenue Leakage — Executive Summary")
    st.caption(
        "Automated detection across 100K+ orders · "
        "Updated on each pipeline run"
    )
    st.divider()

    render_kpi_cards(kpis)
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        render_revenue_trend(data["monthly"])
    with col2:
        render_leakage_breakdown(kpis)

    st.divider()
    render_top_categories(data["category"])