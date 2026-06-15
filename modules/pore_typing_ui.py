import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import plotly.express as px
from modules.pore_model import load_model_by_name
from modules.rac import train_rca_model, predict_k
def render_rca_panel(df_plot, rca_models, key_prefix):
    
    fig_rca = go.Figure()

    # ===============================
    # 散点
    # ===============================
    for t in sorted(df_plot["PoreType"].unique()):
        group = df_plot[df_plot["PoreType"] == t]

        fig_rca.add_trace(
            go.Scatter(
                x=group["CPOR_clean"],
                y=group["CKH_clean"],
                mode="markers",
                marker=dict(size=3, opacity=0.4),
                name=f"Type {t}"
            )
        )

    # ===============================
    # RCA曲线
    # ===============================
    phi_curve = np.linspace(0.01, 0.4, 200)

    for t, params in rca_models.items():

        k_curve = params["a"] * (phi_curve ** params["b"])

        fig_rca.add_trace(
            go.Scatter(
                x=phi_curve,
                y=k_curve,
                mode="lines",
                line=dict(width=2),
                name=f"Type {t}"
            )
        )

    # ===============================
    # ✅ 当前选择高亮（重点🔥）
    # ===============================
    phi_input = st.session_state["phi_input"]
    type_input = st.session_state["type_input"]

    k_pred = rca_models[type_input]["a"] * (phi_input ** rca_models[type_input]["b"])

    fig_rca.add_trace(
        go.Scatter(
            x=[phi_input],
            y=[k_pred],
            mode="markers",
            marker=dict(size=10, color="black"),
            name="Prediction",
        )
    )

    fig_rca.update_layout(
        xaxis_title="CPOR_clean (v/v)",
        yaxis_title="CKH_clean (mD)",
        yaxis=dict(type="log"),
        xaxis=dict(range=[0, 0.4])
    )

    st.plotly_chart(
        fig_rca,
        use_container_width=True,
        key=f"rca_chart_{key_prefix}"
    )

    # ✅ 输出预测
    st.success(f"Predicted k = {k_pred:.2f} mD")


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
                    range=[-2, 4],
                    tickvals=[0.1, 1, 10, 100, 1000, 10000],
                    ticktext=["0.1", "1", "10", "100", "1000", "10000"],
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig_pc, use_container_width=True)

        # =========================
        # Reservoir Classification Tabs
        # =========================
        st.markdown("---")
        st.header("Reservoir Classification Methods")
        # ===============================
        # RCA模型（全局）
        # ===============================
        rca_models = train_rca_model(df_plot)
        # ===============================
        # RCA全局输入
        # ===============================
        st.subheader("RCA Prediction Control")

        col1, col2 = st.columns(2)

        with col1:
            st.session_state["phi_input"] = st.number_input(
                "Input Porosity (v/v)",
                min_value=0.01,
                max_value=0.4,
                value=st.session_state.get("phi_input", 0.15),
                step=0.01
            )

        with col2:
            st.session_state["type_input"] = st.selectbox(
                "Select Pore Type",
                options=sorted(rca_models.keys()),
                index=0
            )

        tab1, tab2, tab3 = st.tabs(["FZI Method", "R35 Method", "Pittman Method"])
        
        # 统一散点样式
        marker_style = dict(
            size=5,
            color=df_plot["PoreType"],
            colorscale="Viridis",
            opacity=0.7
        )

        phi = np.linspace(0.01, 0.4, 100)
        
        # ===============================
        # 数据过滤
        # ===============================
        df_plot = df_plot[df_plot["CKH_clean"] > 0]
        df_plot = df_plot[df_plot["CPOR_clean"] > 0]

        with tab1:

            st.subheader("FZI Method")

            fig = go.Figure()

            # ===============================
            # 散点（按Type）
            # ===============================
            types = sorted(df_plot["PoreType"].dropna().unique())

            color_map = {
                1: "blue", 2: "green", 3: "goldenrod", 4: "orange",
                5: "red", 6: "purple", 7: "brown", 8: "black"
            }

            for t in types:
                group = df_plot[df_plot["PoreType"] == t]

                fig.add_trace(
                    go.Scatter(
                        x=group["CPOR_clean"],
                        y=group["CKH_clean"],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=color_map.get(t, "gray"),
                            opacity=0.6
                        ),
                        name=f"Type {t}"
                    )
                )

            # ===============================
            # 必须补上的变量
            # ===============================
            phi = np.linspace(0.01, 0.35, 200)
            fzi_values = [0.35, 0.8, 1.77, 4]

            ymax = 10**4
            ymin = 10**-3

            # ===============================
            # FZI曲线
            # ===============================
            for fzi in fzi_values:

                k = 1014 * (fzi**2) * (phi**3) / ((1 - phi)**2)

                fig.add_trace(
                    go.Scatter(
                        x=phi,
                        y=k,
                        mode="lines",
                        line=dict(color="gray", width=2),
                        showlegend=False
                    )
                )

                # x 在最右端
                x_end = phi[-1]

                # y 强制限制在可见范围
                y_end = max(min(k[-1], ymax), ymin)

                
                fig.add_annotation(
                    x=x_end,
                    y=y_end,
                    text=f"FZI = {fzi}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=11)
                )
                
            # ===============================
            # 坐标轴
            # ===============================
            fig.update_layout(
                xaxis_title="CPOR_clean (v/v)",
                yaxis_title="CKH_clean (mD)",
                
                xaxis=dict(
                        range=[0, 0.4]  
                    ),

                yaxis=dict(
                    type="log",
                    range=[-3, 4]
                ),

                plot_bgcolor="white",
                legend=dict(title="Pore Type")
            )

            fig.update_xaxes(showgrid=True, gridcolor="lightgray")
            fig.update_yaxes(showgrid=True, gridcolor="lightgray")

            st.plotly_chart(fig, use_container_width=True)
            render_rca_panel(df_plot, rca_models, key_prefix="tab1")

        with tab2:

            st.subheader("R35 Method")

            fig_r35 = go.Figure()

            # 散点
            types = sorted(df_plot["PoreType"].dropna().unique())

            color_map = {
                1: "blue", 2: "green", 3: "goldenrod", 4: "orange",
                5: "red", 6: "purple", 7: "brown", 8: "black"
            }

            for t in types:
                group = df_plot[df_plot["PoreType"] == t]

                fig_r35.add_trace(
                    go.Scatter(
                        x=group["CPOR_clean"],
                        y=group["CKH_clean"],
                        mode="markers",
                        marker=dict(size=3, color=color_map.get(t, "gray"), opacity=0.6),
                        name=f"Type {t}"
                    )
                )

            # R35线（灰色实线 + 标注）
            phi = np.linspace(0.01, 0.35, 200)
            r35_values = [10, 50, 100, 500]

            for r35 in r35_values:

                ptr = r35 * phi / (1 - phi)

                fig_r35.add_trace(
                    go.Scatter(
                        x=phi,
                        y=ptr,
                        mode="lines",
                        line=dict(color="gray", width=2),
                        showlegend=False
                    )
                )

                fig_r35.add_annotation(
                    x=phi[-1],
                    y=ptr[-1],
                    text=f"R35 = {r35}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=10)
                )

            fig_r35.update_layout(
                xaxis_title="CPOR_clean (v/v)",
                yaxis_title="CKH_clean (mD)",
                yaxis=dict(type="log", range=[-3, 4]),
                plot_bgcolor="white",
                legend=dict(title="Pore Type")
            )

            fig_r35.update_xaxes(showgrid=True, gridcolor="lightgray")
            fig_r35.update_yaxes(showgrid=True, gridcolor="lightgray")

            st.plotly_chart(fig_r35, use_container_width=True)
            render_rca_panel(df_plot, rca_models, key_prefix="tab2")

        with tab3:

            st.subheader("Pittman Method")

            fig_pit = go.Figure()

            # 散点
            types = sorted(df_plot["PoreType"].dropna().unique())

            color_map = {
                1: "blue", 2: "green", 3: "goldenrod", 4: "orange",
                5: "red", 6: "purple", 7: "brown", 8: "black"
            }

            for t in types:
                group = df_plot[df_plot["PoreType"] == t]

                fig_pit.add_trace(
                    go.Scatter(
                        x=group["CPOR_clean"],
                        y=group["CKH_clean"],
                        mode="markers",
                        marker=dict(size=3, color=color_map.get(t, "gray"), opacity=0.6),
                        name=f"Type {t}"
                    )
                )

            # ===============================
            # Pittman 曲线
            # ===============================
            phi = np.linspace(0.01, 0.35, 200)
            pit_levels = [0.2, 0.5, 1, 2, 5, 10]

            ymax = 10**4
            ymin = 10**-3

            # 只计算一次 scale（关键）
            k_mean = df_plot["CKH_clean"].median()
            phi_mean = df_plot["CPOR_clean"].median()

            # 防止异常
            if phi_mean <= 0 or phi_mean >= 1:
                phi_mean = 0.15

            base = (phi_mean**3) / ((1 - phi_mean)**2)
            scale = k_mean / base

            # ===============================
            # 循环画曲线
            # ===============================
            for p in pit_levels:

                # 正确公式（必须有）
                k = scale * p * (phi**3) / ((1 - phi)**2)

                fig_pit.add_trace(
                    go.Scatter(
                        x=phi,
                        y=k,
                        mode="lines",
                        line=dict(color="gray", width=2),
                        showlegend=False
                    )
                )

                # 找图内可见的“最右点”
                visible_idx = np.where((k >= ymin) & (k <= ymax))[0]

                if len(visible_idx) == 0:
                    continue

                idx = visible_idx[-1]

                fig_pit.add_annotation(
                    x=phi[idx],
                    y=k[idx],
                    text=f"{p}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10)
                )

            fig_pit.update_layout(
                xaxis_title="CPOR_clean (v/v)",
                yaxis_title="CKH_clean (mD)",
                xaxis=dict(range=[0, 0.4]),
                yaxis=dict(type="log", range=[-3, 4]),
                plot_bgcolor="white",
                legend=dict(title="Pore Type")
            )

            fig_pit.update_xaxes(showgrid=True, gridcolor="lightgray")
            fig_pit.update_yaxes(showgrid=True, gridcolor="lightgray")

            st.plotly_chart(fig_pit, use_container_width=True)
            render_rca_panel(df_plot, rca_models, key_prefix="tab3")
        

