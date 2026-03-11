import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Petroleum Data Analysis with AI", layout="wide")

st.title("🛢 Petroleum Data Analysis with AI")

st.write("Upload petroleum production data (CSV or Excel) to visualize and analyze it.")

file = st.file_uploader("Upload dataset", type=["csv","xlsx"])

if file:
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("Data Preview")
    st.dataframe(df)

    numeric = df.select_dtypes(include="number").columns

    if len(numeric) >= 2:
        x = st.selectbox("X axis", numeric)
        y = st.selectbox("Y axis", numeric, index=1)

        fig = px.scatter(df, x=x, y=y, title="Data Visualization")
        st.plotly_chart(fig, use_container_width=True)

        st.success("AI Insight: Visual trend generated from dataset.")
