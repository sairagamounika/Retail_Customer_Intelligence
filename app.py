"""
Retail Customer Intelligence — Dashboard

Shows segments, CLV, and churn risk in one place for retention and marketing.

Run after notebooks 01–04:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"

st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Retail Customer Intelligence")
st.markdown(
    "One view of your customers: **segments**, **lifetime value**, and **churn risk** for retention and marketing."
)


@st.cache_data
def load_data():
    out = {}
    if (DATA_DIR / "customer_segments_with_clv.csv").exists():
        out["customers"] = pd.read_csv(DATA_DIR / "customer_segments_with_clv.csv")
    elif (DATA_DIR / "customer_segments.csv").exists():
        out["customers"] = pd.read_csv(DATA_DIR / "customer_segments.csv")
    else:
        out["customers"] = None

    if (DATA_DIR / "clv_by_segment.csv").exists():
        out["clv_by_segment"] = pd.read_csv(DATA_DIR / "clv_by_segment.csv")
    else:
        out["clv_by_segment"] = None

    if (DATA_DIR / "customer_churn_risk.csv").exists():
        out["churn_risk"] = pd.read_csv(DATA_DIR / "customer_churn_risk.csv")
    else:
        out["churn_risk"] = None

    if (DATA_DIR / "customer_retention_policy.csv").exists():
        out["retention_policy"] = pd.read_csv(DATA_DIR / "customer_retention_policy.csv")
    else:
        out["retention_policy"] = None

    return out


data = load_data()
customers = data["customers"]
clv_by_segment = data["clv_by_segment"]
churn_risk = data["churn_risk"]
retention_policy = data["retention_policy"]

if customers is None:
    st.warning(
        "No customer/segment data found. Run notebooks **01 → 02 → 03** and save outputs to `data/processed/`."
    )
    st.stop()

# Merge retention policy if available (includes churn_risk + priority_score + actions)
if retention_policy is not None and "customer_id" in customers.columns:
    customers = customers.merge(
        retention_policy[["customer_id", "churn_risk", "priority_score", "recommended_action"]],
        on="customer_id",
        how="left"
    )
    has_retention_policy = True
    has_churn = True
elif churn_risk is not None and "customer_id" in customers.columns:
    customers = customers.merge(
        churn_risk[["customer_id", "churn_risk"]], on="customer_id", how="left"
    )
    has_retention_policy = False
    has_churn = True
else:
    has_retention_policy = False
    has_churn = False

# ----- Overview -----
st.header("Overview")
seg_col = "cluster_name" if "cluster_name" in customers.columns else "cluster"
if seg_col in customers.columns:
    n_segments = customers[seg_col].nunique()
    n_customers = len(customers)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total customers", f"{n_customers:,}")
    with c2:
        st.metric("Segments", n_segments)
    with c3:
        if "clv_12m" in customers.columns:
            st.metric("Total 12‑month CLV (est.)", f"£{customers['clv_12m'].sum():,.0f}")
        else:
            st.metric("Segments", "—")
else:
    st.metric("Total customers", len(customers))

# ----- Segments & CLV -----
st.header("Segments & value")

if clv_by_segment is not None and not clv_by_segment.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CLV by segment")
        disp = clv_by_segment.copy()
        if "avg_clv" in disp.columns:
            disp["avg_clv"] = disp["avg_clv"].round(2)
        if "total_clv" in disp.columns:
            disp["total_clv"] = disp["total_clv"].round(2)
        st.dataframe(disp, use_container_width=True, hide_index=True)
    with col2:
        if "cluster_name" in clv_by_segment.columns and "total_clv" in clv_by_segment.columns:
            st.bar_chart(
                clv_by_segment.set_index("cluster_name")["total_clv"],
                height=300,
            )
else:
    if seg_col in customers.columns:
        st.write("Segment counts:")
        st.write(customers[seg_col].value_counts())
    st.info("Run notebook **03** and save `clv_by_segment.csv` to see CLV by segment.")

# ----- Retention Policy & At-Risk Customers -----
if has_retention_policy and "priority_score" in customers.columns:
    st.header("Retention Policy: Priority Customers")
    st.markdown(
        "**Priority score = churn_risk × CLV** — Focus on customers who are both **at risk** and **valuable** for maximum ROI."
    )
    
    # Show summary by recommended action
    if "recommended_action" in customers.columns:
        action_summary = customers.groupby("recommended_action").agg(
            customers=("customer_id", "count"),
            avg_churn_risk=("churn_risk", "mean"),
            avg_clv=("clv_12m", "mean") if "clv_12m" in customers.columns else ("priority_score", "mean"),
            total_clv=("clv_12m", "sum") if "clv_12m" in customers.columns else ("priority_score", "sum")
        ).sort_values("total_clv", ascending=False)
        
        st.subheader("Summary by Recommended Action")
        st.dataframe(action_summary.style.format({
            "avg_churn_risk": "{:.2%}",
            "avg_clv": "{:,.2f}",
            "total_clv": "{:,.2f}"
        }), use_container_width=True, hide_index=True)
    
    # Filter by action or priority threshold
    st.subheader("Priority Customer List")
    
    filter_option = st.radio(
        "Filter by:",
        ["All customers", "Recommended action", "Priority threshold"],
        horizontal=True
    )
    
    if filter_option == "Recommended action" and "recommended_action" in customers.columns:
        selected_action = st.selectbox(
            "Select action:",
            customers["recommended_action"].dropna().unique()
        )
        priority_list = customers[customers["recommended_action"] == selected_action].copy()
    elif filter_option == "Priority threshold":
        threshold = st.slider(
            "Minimum priority score",
            0.0,
            float(customers["priority_score"].max()) if "priority_score" in customers.columns else 100.0,
            float(customers["priority_score"].quantile(0.9)) if "priority_score" in customers.columns else 50.0,
        )
        priority_list = customers[customers["priority_score"] >= threshold].copy()
    else:
        priority_list = customers.copy()
    
    priority_list = priority_list.sort_values("priority_score", ascending=False)
    
    st.metric("Customers shown", len(priority_list))
    
    display_cols = ["customer_id", "cluster_name" if "cluster_name" in priority_list.columns else seg_col]
    if "churn_risk" in priority_list.columns:
        display_cols.append("churn_risk")
    if "clv_12m" in priority_list.columns:
        display_cols.append("clv_12m")
    if "priority_score" in priority_list.columns:
        display_cols.append("priority_score")
    if "recommended_action" in priority_list.columns:
        display_cols.append("recommended_action")
    
    display_cols = [c for c in display_cols if c in priority_list.columns]
    show = priority_list[display_cols].head(100).copy()
    
    if "churn_risk" in show.columns:
        show["churn_risk"] = (show["churn_risk"] * 100).round(1).astype(str) + "%"
    if "clv_12m" in show.columns:
        show["clv_12m"] = show["clv_12m"].round(2)
    if "priority_score" in show.columns:
        show["priority_score"] = show["priority_score"].round(2)
    
    st.dataframe(show, use_container_width=True, hide_index=True)

elif has_churn and "churn_risk" in customers.columns:
    st.header("Retention: at-risk customers")
    st.markdown(
        "Customers with **high churn risk** are good candidates for retention. "
        "Prioritize those with **high CLV** for maximum impact."
    )
    st.info("💡 Run notebook **04** fully to generate retention policy with priority scores and recommended actions.")
    
    risk_threshold = st.slider(
        "Churn risk threshold (min. to show as at-risk)",
        0.0, 1.0, 0.5, 0.05,
    )
    at_risk = customers[customers["churn_risk"] >= risk_threshold].copy()
    at_risk = at_risk.sort_values("churn_risk", ascending=False)

    if "clv_12m" in at_risk.columns:
        at_risk["priority"] = at_risk["churn_risk"] * at_risk["clv_12m"]
        at_risk = at_risk.sort_values("priority", ascending=False)

    st.metric("Customers above risk threshold", len(at_risk))
    display_cols = ["customer_id", "cluster_name" if "cluster_name" in at_risk.columns else seg_col, "churn_risk"]
    if "clv_12m" in at_risk.columns:
        display_cols.append("clv_12m")
    display_cols = [c for c in display_cols if c in at_risk.columns]
    show = at_risk[display_cols].head(100).copy()
    if "churn_risk" in show.columns:
        show["churn_risk"] = (show["churn_risk"] * 100).round(1).astype(str) + "%"
    if "clv_12m" in show.columns:
        show["clv_12m"] = show["clv_12m"].round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)
else:
    st.header("Retention: at-risk customers")
    st.info("Run notebook **04** and save `customer_churn_risk.csv` to see churn risk and retention priorities.")

st.divider()
st.caption("Built from the Retail Customer Intelligence pipeline (notebooks 01–04).")
