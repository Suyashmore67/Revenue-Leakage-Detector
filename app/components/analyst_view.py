# app/components/analyst_view.py
# PURPOSE: Analyst drill-down view — anomaly timeline,
# category heatmap, and interactive RCA panel.

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render_detector_scorecard(eval_df: pd.DataFrame):
    """Show precision/recall for each detector."""
    st.subheader("Detector Performance")

    cols = st.columns(4)
    colors = {
        "L1_discount_abuse":  "#7F77DD",
        "L2_refund_spike":    "#1D9E75",
        "L3_silent_churn":    "#D85A30",
        "L4_short_lifecycle": "#378ADD",
    }
    labels = {
        "L1_discount_abuse":  "Discount Abuse",
        "L2_refund_spike":    "Refund Spike",
        "L3_silent_churn":    "Silent Churn",
        "L4_short_lifecycle": "Short Lifecycle",
    }

    for i, (_, row) in enumerate(eval_df.iterrows()):
        with cols[i % 4]:
            recall_pct = round(row["recall"] * 100, 1)
            color      = colors.get(row["detector"], "#888")
            label      = labels.get(row["detector"], row["detector"])

            st.markdown(
                f"""
                <div style="
                    border:0.5px solid {color}40;
                    border-radius:10px;
                    padding:12px;
                    text-align:center;
                    background:{color}10;
                ">
                  <div style="font-size:11px;color:#888;
                              margin-bottom:4px;">{label}</div>
                  <div style="font-size:24px;font-weight:500;
                              color:{color};">{recall_pct}%</div>
                  <div style="font-size:11px;color:#888;">recall</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_anomaly_timeline(baseline: pd.DataFrame,
                            selected_category: str):
    """Revenue vs baseline for selected category over time."""
    st.subheader(f"Anomaly Timeline — {selected_category}")

    cat_data = (baseline[baseline["category_en"] == selected_category]
                .sort_values("year_week"))

    if cat_data.empty:
        st.info("No data for selected category.")
        return

    fig = go.Figure()

    # Normal band (shaded area)
    fig.add_trace(go.Scatter(
        x    = pd.concat([cat_data["year_week"],
                          cat_data["year_week"].iloc[::-1]]),
        y    = pd.concat([cat_data["upper_bound"],
                          cat_data["lower_bound"].iloc[::-1]]),
        fill      = "toself",
        fillcolor = "rgba(127,119,221,0.08)",
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "Normal band",
        showlegend= True
    ))

    # Baseline mean
    fig.add_trace(go.Scatter(
        x    = cat_data["year_week"],
        y    = cat_data["rolling_mean"],
        mode = "lines",
        name = "Expected (baseline)",
        line = dict(color="#7F77DD", width=1.5, dash="dot")
    ))

    # Actual revenue
    fig.add_trace(go.Scatter(
        x    = cat_data["year_week"],
        y    = cat_data["weekly_revenue"],
        mode = "lines+markers",
        name = "Actual revenue",
        line = dict(color="#1D9E75", width=2),
        marker = dict(size=4)
    ))

    # Anomaly points (red dots)
    anomalies = cat_data[cat_data["is_anomaly"] == True]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x    = anomalies["year_week"],
            y    = anomalies["weekly_revenue"],
            mode = "markers",
            name = "Anomaly flagged",
            marker = dict(color="#E24B4A", size=10, symbol="x")
        ))

    fig.update_layout(
        height        = 350,
        margin        = dict(l=0, r=0, t=10, b=0),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        xaxis         = dict(showgrid=False,
                             tickangle=45,
                             tickfont=dict(size=9)),
        yaxis         = dict(gridcolor="#f0f0f0",
                             tickprefix="R$",
                             tickformat=",.0f"),
        legend        = dict(orientation="h", y=1.12),
        hovermode     = "x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_category_heatmap(baseline: pd.DataFrame):
    """Heatmap: category × week, colored by z-score."""
    st.subheader("Revenue Anomaly Heatmap")
    st.caption("Red = revenue significantly below baseline for that category-week")

    # Pivot to category × week matrix
    # Only top 15 categories by total revenue for readability
    top_cats = (baseline
                .groupby("category_en")["weekly_revenue"]
                .sum()
                .nlargest(15)
                .index
                .tolist())

    pivot = (baseline[baseline["category_en"].isin(top_cats)]
             .pivot_table(index="category_en",
                          columns="year_week",
                          values="z_score",
                          aggfunc="mean"))

    # Only show last 20 weeks to keep it readable
    pivot = pivot.iloc[:, -20:]

    fig = px.imshow(
        pivot,
        color_continuous_scale = [
            [0.0, "#E24B4A"],
            [0.3, "#FAECE7"],
            [0.5, "#F1EFE8"],
            [0.7, "#E1F5EE"],
            [1.0, "#1D9E75"]
        ],
        zmin   = -3,
        zmax   =  3,
        aspect = "auto",
        labels = dict(color="Z-score")
    )

    fig.update_layout(
        height        = 420,
        margin        = dict(l=0, r=0, t=10, b=0),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        xaxis         = dict(tickangle=45, tickfont=dict(size=8)),
        yaxis         = dict(tickfont=dict(size=10)),
        coloraxis_colorbar = dict(title="Z-score", thickness=12)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_rca_panel(rca: pd.DataFrame):
    """Interactive RCA panel — select an anomaly, see root cause."""
    st.subheader("Root Cause Analysis")
    st.caption(
        "Select an anomaly below to see which seller, "
        "region and customer tier drove the revenue drop"
    )

    if rca.empty:
        st.info("No RCA data available.")
        return

    # Display RCA table
    display_cols = ["year_week", "category", "revenue_gap",
                    "z_score", "root_cause_state",
                    "root_cause_seller", "seller_cancel_rate"]

    available = [c for c in display_cols if c in rca.columns]
    rca_display = rca[available].copy()

    if "revenue_gap" in rca_display.columns:
        rca_display["revenue_gap"] = rca_display["revenue_gap"].apply(
            lambda x: f"R${x:,.0f}"
        )

    # Row selector
    selected_idx = st.selectbox(
        "Select anomaly to investigate",
        options  = range(len(rca_display)),
        format_func = lambda i: (
            f"{rca_display.iloc[i]['year_week']} — "
            f"{rca_display.iloc[i]['category']} — "
            f"Gap: {rca_display.iloc[i]['revenue_gap']}"
        )
    )

    # Show selected RCA details
    selected = rca.iloc[selected_idx]

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Root cause region**")
        state = selected.get("root_cause_state", "N/A")
        st.markdown(f"### {state}")
        st.caption("Seller state with highest impact")

    with col2:
        st.markdown("**Problem seller**")
        seller = str(selected.get("root_cause_seller", "N/A"))
        st.markdown(f"### {seller[:12]}...")
        cancel = selected.get("seller_cancel_rate", 0)
        st.caption(f"Cancel rate: {cancel}%")

    with col3:
        st.markdown("**Recommended action**")
        action = selected.get("recommended_action", "Investigate manually")
        st.info(action)

    st.divider()
    st.dataframe(rca_display, use_container_width=True, hide_index=True)


def render(data: dict):
    """Render the full analyst view."""
    st.markdown("### Revenue Leakage — Analyst Drill-Down")
    st.caption(
        "Technical view · anomaly detection results · "
        "root cause breakdown"
    )
    st.divider()

    render_detector_scorecard(data["eval_df"])
    st.divider()

    # Category selector for timeline
    all_cats = sorted(data["baseline"]["category_en"].dropna().unique())
    selected_cat = st.selectbox(
        "Select category to inspect",
        options=all_cats,
        index=0
    )
    render_anomaly_timeline(data["baseline"], selected_cat)
    st.divider()

    render_category_heatmap(data["baseline"])
    st.divider()

    render_rca_panel(data["rca"])