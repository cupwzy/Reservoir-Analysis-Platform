import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.pore_model import load_model


def run():

    st.header("Pore Typing")

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
- Figure: Pore Throat Distribution (Raw Data)
""")

    uploaded_file = st.file_uploader("Upload Data", type=["xlsx"])

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
        model = load_model()

        pred = model.predict(X)
        proba = model.predict_proba(X)

        df["PoreType"] = pred
        df["Confidence"] = proba.max(axis=1)

        # 不确定性判断
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

        st.dataframe(
            df[[
                "wellName",
                "PTR_P",
                "PORE_V_P",
                "PoreType",
                "Confidence",
                "Final_Type"
            ]]
        )

        # =========================
        # 5. 数据准备（画图）
        # =========================
        df_plot = df.copy()

        df_plot["PTR_P"] = pd.to_numeric(df_plot["PTR_P"], errors="coerce")
        df_plot["PORE_V_P"] = pd.to_numeric(df_plot["PORE_V_P"], errors="coerce")

        # 范围过滤
        df_plot = df_plot[
            (df_plot["PTR_P"] > 0) &
            (df_plot["PTR_P"] <= 1000) &
            (df_plot["PORE_V_P"] > 0) &
            (df_plot["PORE_V_P"] <= 20)
        ]

        df_plot = df_plot.dropna(subset=["PTR_P", "PORE_V_P", "PoreType"])

        # log坐标
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

            df_cls = df_plot[df_plot["PoreType"] == cls]

            if len(df_cls) < 5:
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

            # 可选平滑（推荐）
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

        # 高亮不确定点
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

        # =========================
        # 7. 图形样式
        # =========================
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
            legend=dict(
                x=1.02,
                y=1
            ),
            margin=dict(l=50, r=150, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # =========================
        # 新模块：未分类数据预测
        # =========================
        st.markdown("---")
        st.header("New Data Prediction (Unlabeled Data)")

        st.info("""
        Upload new dataset without labels.
        Model will predict pore type automatically and provide confidence.
        """)

        new_file = st.file_uploader("Upload Unlabeled Data", type=["xlsx"], key="new_data")

        if new_file:

            # =========================
            # 1. 读取新数据
            # =========================
            df_new = pd.read_excel(new_file, skiprows=[1])

            st.subheader("Input Data")
            st.dataframe(df_new.head())

            # =========================
            # 2. 特征检查
            # =========================
            feature_cols = [
                "CKH_clean",
                "CPOR_clean",
                "PTR_P",
                "PC_STRESS_CORR",
                "SW_STRESS_CORR"
            ]

            missing = [c for c in feature_cols if c not in df_new.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            df_new = df_new.dropna(subset=feature_cols)

            X_new = df_new[feature_cols]

            # =========================
            # 3. 模型预测
            # =========================
            model = load_model()

            pred_new = model.predict(X_new)
            proba_new = model.predict_proba(X_new)

            df_new["PoreType"] = pred_new
            df_new["Confidence"] = proba_new.max(axis=1)

            # ✅ 不确定性判断
            df_new["Final_Type"] = df_new.apply(
                lambda row: f"Type {row['PoreType']}"
                if row["Confidence"] >= 0.6
                else "Transitional",
                axis=1
            )

            # =========================
            # 4. 输出结果
            # =========================
            st.subheader("Prediction Result")

            st.dataframe(
                df_new[[
                    "wellName",
                    "PTR_P",
                    "PORE_V_P",
                    "PoreType",
                    "Confidence",
                    "Final_Type"
                ]]
            )

            # =========================
            # 5. Figure 5-10（新数据）
            # =========================
            st.subheader("Figure (New Data)")

            df_plot_new = df_new.copy()

            df_plot_new["PTR_P"] = pd.to_numeric(df_plot_new["PTR_P"], errors="coerce")
            df_plot_new["PORE_V_P"] = pd.to_numeric(df_plot_new["PORE_V_P"], errors="coerce")

            # 同样过滤
            df_plot_new = df_plot_new[
                (df_plot_new["PTR_P"] > 0) &
                (df_plot_new["PTR_P"] <= 1000) &
                (df_plot_new["PORE_V_P"] > 0) &
                (df_plot_new["PORE_V_P"] <= 20)
            ]

            df_plot_new = df_plot_new.dropna(subset=["PTR_P", "PORE_V_P"])

            df_plot_new["log_PTR"] = np.log10(df_plot_new["PTR_P"])

            fig_new = go.Figure()

            for i, cls in enumerate(sorted(df_plot_new["PoreType"].unique())):

                df_cls = df_plot_new[df_plot_new["PoreType"] == cls]

                if len(df_cls) < 5:
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

                fig_new.add_trace(
                    go.Scatter(
                        x=10**grouped["log_PTR"],
                        y=grouped["PORE_V_P"],
                        mode="lines",
                        name=f"Type {cls}"
                    )
                )

            fig_new.update_layout(
                xaxis=dict(type="log"),
                yaxis=dict(range=[0, 20]),
                margin=dict(l=40, r=120, t=40, b=40)
            )

            st.plotly_chart(fig_new, use_container_width=True)

            # =========================
            # 6. 下载结果
            # =========================
            st.subheader("Download Results")

            csv = df_new.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="pore_typing_results.csv",
                mime="text/csv"
            )