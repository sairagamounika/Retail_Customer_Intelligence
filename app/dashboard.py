
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Config
st.set_page_config(page_title="Retail Customer Retention", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
POLICY_PATH = DATA_DIR / "customer_retention_policy.csv"

@st.cache_data
def load_data():
    if POLICY_PATH.exists():
        return pd.read_csv(POLICY_PATH)
    return None

df = load_data()

st.title("Customer Retention Dashboard")

if df is not None:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    
    avg_churn_risk = df["churn_risk"].mean()
    col2.metric("Avg Churn Risk", f"{avg_churn_risk:.1%}")
    
    high_risk_count = len(df[df["churn_risk"] > 0.7])
    col3.metric("High Risk (>70%)", f"{high_risk_count:,}")
    
    total_clv = df["clv_12m"].sum()
    col4.metric("Total 12M CLV", f"£{total_clv:,.0f}")
    
    st.markdown("---")
    
    # Segment Analysis
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Risk & Value by Segment")
        fig = px.scatter(
            df, 
            x="clv_12m", 
            y="churn_risk", 
            color="cluster_name",
            size="priority_score",
            hover_data=["customer_id", "recommended_action"],
            log_x=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_chart2:
        st.subheader("Action Distribution")
        action_counts = df["recommended_action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig2 = px.bar(action_counts, x="Count", y="Action", orientation='h', height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Actionable Table
    st.subheader("Priority Retention List")
    
    action_filter = st.multiselect(
        "Filter by Recommended Action",
        options=df["recommended_action"].unique(),
        default=df["recommended_action"].unique()
    )
    
    if action_filter:
        filtered_df = df[df["recommended_action"].isin(action_filter)]
    else:
        filtered_df = df
        
    st.dataframe(
        filtered_df[["customer_id", "cluster_name", "clv_12m", "churn_risk", "priority_score", "recommended_action"]]
        .sort_values("priority_score", ascending=False)
        .style.format({
            "clv_12m": "£{:,.2f}",
            "churn_risk": "{:.1%}",
            "priority_score": "{:.1f}"
        }),
        use_container_width=True
    )
    
    # Download
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered CSV",
        data=csv,
        file_name='retention_priority_list.csv',
        mime='text/csv',
    )

else:
    st.error(f"Data not found at {POLICY_PATH}. Please run the notebooks (specifically notebook 04) to generate the retention policy data.")
