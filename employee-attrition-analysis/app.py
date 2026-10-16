import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_data

st.set_page_config(page_title="HR Attrition Analytics Dashboard", layout="wide")

st.title("📊 HR Analytics & Employee Attrition Dashboard")
st.markdown("Real-time exploratory analytics to monitor turnover risk factors across departments and roles.")

# Load Data
@st.cache_data
def get_dashboard_data():
    return load_data()

df = get_dashboard_data()

# KPI Metrics
total_employees = len(df)
attrition_count = (df['Attrition'] == 'Yes').sum()
attrition_rate = (attrition_count / total_employees) * 100
avg_monthly_income = df['MonthlyIncome'].mean()
avg_tenure = df['YearsAtCompany'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Headcount", f"{total_employees}")
col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
col3.metric("Avg Monthly Salary", f"${avg_monthly_income:,.0f}")
col4.metric("Avg Tenure", f"{avg_tenure:.1f} Years")

st.divider()

# Charts Grid
c1, c2 = st.columns(2)

with c1:
    fig_dept = px.histogram(df, x="Department", color="Attrition", barmode="group",
                            title="Attrition by Department", color_discrete_sequence=["#2b5c8f", "#d9534f"])
    st.plotly_chart(fig_dept, use_container_width=True)

with c2:
    fig_salary = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition",
                        title="Monthly Income Distribution vs Attrition", color_discrete_sequence=["#2b5c8f", "#d9534f"])
    st.plotly_chart(fig_salary, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    fig_overtime = px.histogram(df, x="OverTime", color="Attrition", barmode="group",
                                title="Impact of OverTime on Attrition", color_discrete_sequence=["#2b5c8f", "#d9534f"])
    st.plotly_chart(fig_overtime, use_container_width=True)

with c4:
    fig_age = px.histogram(df, x="Age", color="Attrition", nbins=20,
                           title="Age Distribution vs Resignation", color_discrete_sequence=["#2b5c8f", "#d9534f"])
    st.plotly_chart(fig_age, use_container_width=True)