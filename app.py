import streamlit as st
from modules.uploader import load_master_table
from modules.mapping import plot_well_map
from modules.well_data import load_and_merge_well_data
import pandas as pd
import plotly.express as px
from modules import pore_typing_ui

# ===============================
# 页面配置
# ===============================
st.set_page_config(layout="wide")

st.title("Reservoir Analysis Platform")

# ✅ 更新 Navigation
menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Well Map",
        "Multi-Well Analysis",   # ✅ 原 Workspace
        "Pore Typing"            # ✅ 新增模块
    ]
)


# ===============================
# Session State
# ===============================
if "master_df" not in st.session_state:
    st.session_state.master_df = None

# =========================
# HOME
# =========================
if menu == "Home":

    st.subheader("Platform Overview")

    st.write("""
    This platform supports:

    - Well mapping
    - Multi-well data integration
    - Reservoir property analysis
    - Net pay calculation
    """)

    st.markdown("---")

    # =========================
    # Data Status
    # =========================
    st.markdown("### Data Status")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.master_df is not None:
            st.success("Master Table Loaded")
            st.write(f"Wells: {st.session_state.master_df['Name'].nunique()}")
        else:
            st.warning("Master Table Not Loaded")

    with col2:
        if "df_all" in st.session_state:
            st.success("Well Data Loaded")
            st.write(f"Total Wells: {st.session_state.df_all['Well'].nunique()}")
        else:
            st.warning("Well Data Not Loaded")

    st.markdown("---")

    # =========================
    # Preview
    # =========================
    if st.session_state.master_df is not None:
        st.markdown("### Well Overview")
        st.dataframe(st.session_state.master_df.head(), use_container_width=True)

    # =========================
    # Workflow
    # =========================
    st.markdown("### Workflow")

    st.info("""
    1. Upload Master Table (Well Map)
    2. Upload Well Data (Workspace)
    3. Select ZONE
    4. Perform Analysis
    """)

# =========================
# WELL MAP
# =========================
elif menu == "Well Map":
    st.subheader("Well Location Map")

    st.markdown("""
    ### Input Data Requirements

    The uploaded Excel file must contain the following columns:

    - Name: Well name  
    - Well symbol: Oil or Injection  
    - Surface X: X coordinate  
    - Surface Y: Y coordinate  
    - Target: Reservoir target zone  
    """)


    file = st.file_uploader("Upload Master Table", type=["xlsx"])

    # ✅ 如果用户上传 → 存入 session
    
    if file is not None:
        df = load_master_table(file)
        st.session_state.master_df = df

    # ✅ 永远从 session 读取
    if st.session_state.master_df is not None:
        df = st.session_state.master_df

        st.success("Master table loaded")
        st.dataframe(df, use_container_width=True)

        fig = plot_well_map(df)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please upload master table.")


# =========================
# Multi-Well Analysis（原 Workspace）
# =========================
elif menu == "Multi-Well Analysis":

    st.subheader("Multi-Well Analysis")
    st.info("This module is used for multi-well data integration and comparison")

    # =========================
    # Upload
    # =========================
    files = st.file_uploader(
        "Upload Well Interpretation Files",
        type=["xlsx"],
        accept_multiple_files=True,
        key="well_uploader"
    )

    # ✅ 只负责“存”
    if files:
        try:
            from modules.well_data import load_and_merge_well_data
            df_all = load_and_merge_well_data(files)

            st.session_state.df_all = df_all
            st.success(f"Loaded {df_all['Well'].nunique()} wells")

        except Exception as e:
            st.error(str(e))

    # =========================
    # 主逻辑（永远从 session 读）
    # =========================
    if "df_all" not in st.session_state:
        st.warning("Please upload well data.")
        st.stop()

    df_all = st.session_state.df_all

    # =========================
    # ZONE Selection
    # =========================
    st.markdown("### ZONE Selection")

    zones = sorted(df_all["ZONE"].dropna().unique())

    selected_zones = st.multiselect(
        "Select ZONE (optional)",
        options=zones,
        default=[],
        key="zone_selector"
    )

    # =========================
    # Filter
    # =========================
    if selected_zones:
        df_filtered = df_all[df_all["ZONE"].isin(selected_zones)]
    else:
        df_filtered = df_all
    
    # ✅ 先做类型转换（必须）
    numeric_cols = ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"]

    for col in numeric_cols:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    # =========================
    # Combined Data
    # =========================
    st.markdown("### Combined Data")
    st.dataframe(df_filtered, use_container_width=True)

    # =========================
    # Summary
    # =========================
    st.markdown("### Summary")

    required_cols = ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"]

    # ✅ 条件1：字段存在
    if not all(col in df_filtered.columns for col in required_cols):
        st.info("Summary not available: missing required columns.")

    else:
        df_calc = df_filtered.dropna(subset=required_cols)

        # ✅ 条件2：有有效数据
        if df_calc.empty:
            st.info("Summary not available: no valid data after filtering.")

        else:
            md_total = df_calc["MD_THK"].sum()
            tvd_total = df_calc["TVDSS_THK"].sum()

            # ✅ 条件3：厚度合理
            if tvd_total <= 0:
                st.info("Summary not available: total thickness is zero.")

            else:
                # ✅ 正常计算
                vsh = (df_calc["VSH"] * df_calc["TVDSS_THK"]).sum() / tvd_total
                phie = (df_calc["PHIE"] * df_calc["TVDSS_THK"]).sum() / tvd_total
                swe = (df_calc["SWE"] * df_calc["TVDSS_THK"]).sum() / tvd_total

                import pandas as pd

                summary_df = pd.DataFrame({
                    "Metric": ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"],
                    "Value": [md_total, tvd_total, vsh, phie, swe]
                })

                st.dataframe(summary_df, use_container_width=True)


    # =========================
    # Plotly
    # =========================

    st.markdown("### Thickness Comparison (ZONE + Total)")

    # =========================
    # ZONE级别
    # =========================
    df_zone = df_filtered.groupby(
        ["Well", "ZONE"], as_index=False
    )["TVDSS_THK"].sum()

    # =========================
    # Total（关键）
    # =========================
    df_total = df_zone.groupby("Well", as_index=False)["TVDSS_THK"].sum()
    df_total["ZONE"] = "Total"

    # =========================
    # 合并
    # =========================
    import pandas as pd

    df_plot = pd.concat([df_zone, df_total], ignore_index=True)

    # =========================
    # Plot
    # =========================
    import plotly.express as px

    fig_thk = px.bar(
        df_plot,
        x="Well",
        y="TVDSS_THK",
        color="ZONE",
        barmode="group",   # ✅ 对比关键
        title="Thickness by Well and ZONE"
    )

    st.plotly_chart(fig_thk, use_container_width=True)


    st.markdown("### Reservoir Properties (ZONE + Total)")

    def weighted_avg(g, col):
        if g["TVDSS_THK"].sum() == 0:
            return 0
        return (g[col] * g["TVDSS_THK"]).sum() / g["TVDSS_THK"].sum()
    
    # =========================
    # ZONE级别计算
    # =========================
    df_prop_zone = df_filtered.groupby(["Well", "ZONE"]).apply(
        lambda g: pd.Series({
            "PHIE": weighted_avg(g, "PHIE"),
            "VSH": weighted_avg(g, "VSH"),
            "SWE": weighted_avg(g, "SWE")
        })
    ).reset_index()

    # =========================
    # Total计算
    # =========================
    df_prop_total = df_filtered.groupby("Well").apply(
        lambda g: pd.Series({
            "PHIE": weighted_avg(g, "PHIE"),
            "VSH": weighted_avg(g, "VSH"),
            "SWE": weighted_avg(g, "SWE")
        })
    ).reset_index()

    df_prop_total["ZONE"] = "Total"

    # =========================
    # 合并
    # =========================
    df_prop_all = pd.concat([df_prop_zone, df_prop_total], ignore_index=True)

    # =========================
    # 转长格式
    # =========================
    df_melt = df_prop_all.melt(
        id_vars=["Well", "ZONE"],
        value_vars=["PHIE", "VSH", "SWE"],
        var_name="Property",
        value_name="Value"
    )

    # =========================
    # Plot
    # =========================
    fig_prop = px.bar(
        df_melt,
        x="Well",
        y="Value",
        color="ZONE",            # ✅ 用ZONE区分
        facet_col="Property",    # ✅ 分属性展示
        barmode="group",
        title="Reservoir Properties by Well, ZONE and Total"
    )

    st.plotly_chart(fig_prop, use_container_width=True)

# Pore Typing（独立模块）
elif menu == "Pore Typing":

    st.header("Pore Typing (MICP + Machine Learning)")

    st.info("MICP-based pore structure classification using Machine Learning")

    st.markdown("---")

    # 调用模块
    pore_typing_ui.run()




