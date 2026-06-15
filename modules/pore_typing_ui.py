import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import plotly.express as px
from modules.pore_model import load_model_by_name

def run():

    # ===============================
    # Model Selection
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
    # 说明
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
    # 唯一一个 Upload Data
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

        # 关键修复
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

        # Transitional点
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
        # Capillary Pressure 曲线
        # =========================
        if "SW_STRESS_CORR" in df_plot.columns and "PC_STRESS_CORR" in df_plot.columns:

            st.subheader("Capillary Pressure Curves")

            # 一定要 copy，避免污染原数据
            df_pc = df_plot.copy()

            # =========================
            # 强制清洗 Sw 和 Pc
            # =========================
            df_pc["SW_STRESS_CORR"] = pd.to_numeric(
                df_pc["SW_STRESS_CORR"], errors="coerce"
            )

            df_pc["PC_STRESS_CORR"] = pd.to_numeric(
                df_pc["PC_STRESS_CORR"], errors="coerce"
            )

            # 删除 NaN
            df_pc = df_pc.dropna(subset=["SW_STRESS_CORR", "PC_STRESS_CORR"])

            # 过滤 Sw（0–1）
            df_pc = df_pc[
                (df_pc["SW_STRESS_CORR"] >= 0) &
                (df_pc["SW_STRESS_CORR"] <= 1)
            ]

            # 过滤 Pc
            df_pc = df_pc[
                (df_pc["PC_STRESS_CORR"] > 0) &
                (df_pc["PC_STRESS_CORR"] < 1e6)
            ]

            # debug
            st.write("Sw range:", df_pc["SW_STRESS_CORR"].min(), df_pc["SW_STRESS_CORR"].max())

            fig_pc = go.Figure()

            for cls in sorted(df_pc["PoreType"].dropna().unique()):

                group = df_pc[df_pc["PoreType"] == cls].copy()

                if len(group) < 5:
                    continue

                group = group.sort_values("SW_STRESS_CORR")

                fig_pc.add_trace(
                    go.Scatter(
                        x=group["SW_STRESS_CORR"],
                        y=group["PC_STRESS_CORR"],
                        mode="lines",
                        name=f"Type {cls}",
                        opacity=0.6
                    )
                )

            fig_pc.update_layout(
                xaxis_title="Water Saturation (Sw)",
                yaxis_title="Capillary Pressure (Pc)",
                yaxis=dict(
                    type="log",
                    tickvals=[1, 10, 100, 1000, 10000],
                    ticktext=["1", "10", "100", "1000", "10000"],
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig_pc, use_container_width=True)

        # =========================
        # Reservoir Classification Tabs
        # =========================
        st.markdown("---")
        st.header("Reservoir Classification Methods")

        tab1, tab2, tab3 = st.tabs(["FZI Method", "R35 Method", "Pittman Method"])
        
        # 统一散点样式
        marker_style = dict(
            size=5,
            color=df_plot["PoreType"],
            colorscale="Viridis",
            opacity=0.7
        )

        phi = np.linspace(0.01, 0.4, 100)

        with tab1:

            st.subheader("FZI Method")

            df_fzi = df_plot.copy()

            # =========================
            # 1. 计算 FZI（核心）
            # =========================
            df_fzi["RQI"] = 0.0314 * np.sqrt(
                df_fzi["CKH_clean"] / df_fzi["CPOR_clean"]
            )

            df_fzi["phi_z"] = df_fzi["CPOR_clean"] / (1 - df_fzi["CPOR_clean"])

            df_fzi["FZI"] = df_fzi["RQI"] / df_fzi["phi_z"]

            # =========================
            # 2. FZI 分级规则
            # =========================
            def classify_fzi(fzi):

                if fzi < 0.35:
                    return "Poor"
                elif fzi < 0.8:
                    return "Low"
                elif fzi < 1.77:
                    return "Medium"
                elif fzi < 4:
                    return "Good"
                else:
                    return "Excellent"

            df_fzi["FlowUnit"] = df_fzi["FZI"].apply(classify_fzi)

            # =========================
            # 3. Flow Unit 颜色映射（专业）
            # =========================
            color_map = {
                "Poor": "blue",
                "Low": "green",
                "Medium": "yellow",
                "Good": "orange",
                "Excellent": "red"
            }

            # =========================
            # 4. 绘图（按 Flow Unit 分类）
            # =========================
            fig_fzi = go.Figure()

            for fu in color_map.keys():

                group = df_fzi[df_fzi["FlowUnit"] == fu]

                if len(group) == 0:
                    continue

                fig_fzi.add_trace(
                    go.Scatter(
                        x=group["CPOR_clean"],
                        y=group["CKH_clean"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=color_map[fu]
                        ),
                        name=fu
                    )
                )

            # =========================
            # 5. FZI 等值线（参考线）
            # =========================
            phi = np.linspace(0.01, 0.4, 100)

            fzi_values = [0.35, 0.8, 1.77, 4]

            for fzi in fzi_values:

                k = (fzi * phi / (1 - phi))**2

                fig_fzi.add_trace(
                    go.Scatter(
                        x=phi,
                        y=k,
                        mode="lines",
                        line=dict(color="black", dash="dot"),
                        name=f"FZI={fzi}"
                    )
                )

            # =========================
            # 6. 图形样式
            # =========================
            fig_fzi.update_layout(
                xaxis_title="Porosity (φ)",
                yaxis_title="Permeability (k)",
                yaxis=dict(
                    type="log",
                    tickvals=[1, 10, 100, 1000, 10000],
                    ticktext=["1", "10", "100", "1000", "10000"]
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig_fzi, use_container_width=True)

            # =========================
            # 7. 输出统计信息
            # =========================
            st.write("FZI Summary")

            st.dataframe(
                df_fzi[["FZI", "FlowUnit"]].describe()
            )
        with tab2:

            st.subheader("R35 Method")

            fig_r35 = go.Figure()

            fig_r35.add_trace(
                go.Scatter(
                    x=df_plot["CPOR_clean"],
                    y=df_plot["CKH_clean"],
                    mode="markers",
                    marker=marker_style,
                    name="Samples"
                )
            )

            r35_values = [10, 50, 100, 500]

            for r35 in r35_values:
                ptr = r35 * phi / (1 - phi)
                fig_r35.add_trace(
                    go.Scatter(
                        x=phi,
                        y=ptr,
                        mode="lines",
                        line=dict(color="black", dash="dot"),
                        name=f"R35={r35}"
                    )
                )

            fig_r35.update_layout(
                xaxis_title="Porosity (φ)",
                yaxis_title="Pore Throat Radius (μm)",
                yaxis=dict(type="log")
            )

            st.plotly_chart(fig_r35, use_container_width=True)
        
        with tab3:

            st.subheader("Pittman Method")

            fig_pittman = go.Figure()

            fig_pittman.add_trace(
                go.Scatter(
                    x=df_plot["CPOR_clean"],
                    y=df_plot["CKH_clean"],
                    mode="markers",
                    marker=marker_style,
                    name="Samples"
                )
            )

            pit_levels = [0.2, 0.5, 1, 2, 5, 10]

            for p in pit_levels:
                k = (phi**2.5) * (p * 50)

                fig_pittman.add_trace(
                    go.Scatter(
                        x=phi,
                        y=k,
                        mode="lines",
                        line=dict(color="black"),
                        name=f"{p}"
                    )
                )

            fig_pittman.update_layout(
                xaxis_title="Porosity (φ)",
                yaxis_title="Permeability (k)",
                yaxis=dict(type="log")
            )

            st.plotly_chart(fig_pittman, use_container_width=True)
