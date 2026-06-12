import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import plotly.express as px
from modules.pore_model import load_model_by_name

def run():

    # ===============================
    # ✅ Model Selection
    # ===============================
    if not os.path.exists("models"):
        st.error("Models folder not found")
        return

    st.sidebar.subheader("Model Selection")

    model_files = sorted(
        [f for f in os.listdir("models") if f.endswith(".pkl")]
    )

    if not model_files:
        st.error("No model files found in /models")
        return

    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_files,
        index=0,
        key="model_select_box_main"
    )

    st.header("Pore Typing")

    st.info(f"Current Model: {selected_model}")

    # ===============================
    # ✅ 说明
    # ===============================
    st.info("""
Feature-based pore typing using Random Forest (8 classes)

Input requirements:
- Excel file
- First row: column names
- Second row: units (will be skipped)

Output:
- Predicted Pore Type
- Prediction Confidence
- Transitional Type flag
- Figure: Pore Throat Distribution
""")

    # ===============================
    # ✅ ✅ 唯一一个 Upload Data（重要❗）
    # ===============================
    uploaded_file = st.file_uploader(
        "Upload Analysis Data",
        type=["xlsx"],
        key="main_data_upload"
    )

    if uploaded_file:

        # =========================
        # 1. 读取数据
        # =========================
        df = pd.read_excel(uploaded_file, skiprows=[1])

        # =========================
        # 2. 特征选择
        # =========================
        feature_cols = [
            "CKH_clean",
            "CPOR_clean",
            "PTR_P",
            "PC_STRESS_CORR",
            "SW_STRESS_CORR"
        ]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        df = df.dropna(subset=feature_cols)

        X = df[feature_cols]

        # =========================
        # 3. 模型预测
        # =========================
        model = load_model_by_name(selected_model)

        pred = model.predict(X)
        proba = model.predict_proba(X)

        df["PoreType"] = pred
        df["Confidence"] = proba.max(axis=1)

        df["Final_Type"] = df.apply(
            lambda row: f"Type {row['PoreType']}"
            if row["Confidence"] >= 0.6
            else "Transitional",
            axis=1
        )

        # =========================
        # 4. 输出结果表
        # =========================
        st.subheader("Prediction Result")

        display_cols = [c for c in [
            "wellName", "PTR_P", "PORE_V_P",
            "PoreType", "Confidence", "Final_Type"
        ] if c in df.columns]

        st.dataframe(df[display_cols])

        # =========================
        # 5. 数据准备（画图）
        # =========================
        df_plot = df.copy()

        df_plot["PTR_P"] = pd.to_numeric(df_plot["PTR_P"], errors="coerce")
        df_plot["PORE_V_P"] = pd.to_numeric(df_plot["PORE_V_P"], errors="coerce")

        df_plot = df_plot[
            (df_plot["PTR_P"] > 0) &
            (df_plot["PTR_P"] <= 1000) &
            (df_plot["PORE_V_P"] > 0) &
            (df_plot["PORE_V_P"] <= 20)
        ]

        df_plot = df_plot.dropna(subset=["PTR_P", "PORE_V_P", "PoreType"])

        # ✅ ✅ 关键修复（必须有）
        df_plot = df_plot[df_plot["PTR_P"] > 0]
        df_plot["log_PTR"] = np.log10(df_plot["PTR_P"])

        # =========================
        # 6. Figure 5-10 图
        # =========================
        st.subheader("Figure: Pore Throat Distribution (Raw Data)")

        fig = go.Figure()

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
        ]

        for i, cls in enumerate(sorted(df_plot["PoreType"].unique())):

            df_cls = df_plot[df_plot["PoreType"] == cls].copy()

            if len(df_cls) < 5:
                continue

            if df_cls["log_PTR"].min() == df_cls["log_PTR"].max():
                continue

            df_cls = df_cls.sort_values("log_PTR")

            bins = np.linspace(
                df_cls["log_PTR"].min(),
                df_cls["log_PTR"].max(),
                50
            )

            df_cls["bin"] = np.digitize(df_cls["log_PTR"], bins)

            grouped = df_cls.groupby("bin").agg({
                "log_PTR": "mean",
                "PORE_V_P": "mean"
            }).dropna()

            grouped = grouped.rolling(3, center=True).mean().dropna()

            fig.add_trace(
                go.Scatter(
                    x=10**grouped["log_PTR"],
                    y=grouped["PORE_V_P"],
                    mode="lines",
                    name=f"Type {cls}",
                    line=dict(width=2, color=colors[i % len(colors)])
                )
            )

        # ✅ Transitional点
        df_uncertain = df_plot[df_plot["Confidence"] < 0.6]

        if not df_uncertain.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_uncertain["PTR_P"],
                    y=df_uncertain["PORE_V_P"],
                    mode="markers",
                    name="Transitional",
                    marker=dict(color="black", size=4, opacity=0.5)
                )
            )

        fig.update_layout(
            xaxis=dict(
                title="Pore Throat Radius (μm)",
                type="log",
                tickvals=[0.01, 0.1, 1, 10, 100, 1000]
            ),
            yaxis=dict(
                title="Pore Volume (%)",
                range=[0, 20]
            ),
            legend=dict(x=1.02, y=1),
            margin=dict(l=50, r=150, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # ✅ Capillary Pressure 曲线（新增）
        # =========================
        if "SW_STRESS_CORR" in df_plot.columns and "PC_STRESS_CORR" in df_plot.columns:

            st.subheader("Capillary Pressure (Pc-Sw Curve)")

            df_pc = df_plot.dropna(subset=["SW_STRESS_CORR", "PC_STRESS_CORR"])

            if not df_pc.empty:

                fig_pc = go.Figure()

                fig_pc.add_trace(
                    go.Scatter(
                        x=df_pc["SW_STRESS_CORR"],
                        y=df_pc["PC_STRESS_CORR"],
                        mode="lines",
                        name="Pc-Sw",
                        line=dict(width=2)
                    )
                )

                fig_pc.update_layout(
                    xaxis_title="Water Saturation (Sw)",
                    yaxis_title="Capillary Pressure (Pc)",
                    margin=dict(l=50, r=50, t=50, b=50)
                )

                st.plotly_chart(fig_pc, use_container_width=True)

            else:
                st.warning("No valid Pc-Sw data available for plotting")