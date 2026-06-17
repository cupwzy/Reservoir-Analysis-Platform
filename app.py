import streamlit as st
import pandas as pd
import plotly.express as px

from modules.uploader import load_master_table
from modules.mapping import plot_well_map
from modules.well_data import load_and_merge_well_data
from modules import pore_typing_ui
from modules import autonomous_pore_typing_ui
from modules.ui_theme import apply_theme

# ===============================
# 页面配置
# ===============================
st.set_page_config(layout="wide")
apply_theme()

st.title("Reservoir Analysis Platform")

# ✅ 更新 Navigation
if "navigation" not in st.session_state:
    st.session_state.navigation = "Home"


def _set_navigation(target):
    st.session_state.navigation = target


menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Well Map",
        "Multi-Well Analysis", 
        "Pore Typing",
        "Autonomous Pore Typing"
    ],
    key="navigation"
)

if menu == "Pore Typing":
    pore_typing_ui.run()

if menu == "Autonomous Pore Typing":
    autonomous_pore_typing_ui.run()

# ===============================
# Session State
# ===============================
if "master_df" not in st.session_state:
    st.session_state.master_df = None

# =========================
# HOME
# =========================
if menu == "Home":

    st.markdown(
        """
        <div class="rp-hero">
            <div class="rp-kicker">Reservoir intelligence workspace</div>
            <h2>From well files to pore-system decisions.</h2>
            <p>
                A focused reservoir analysis platform for loading well metadata, comparing multi-well
                interpretation results, and classifying carbonate pore throat systems from capillary-pressure
                and pore throat radius data.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    master_loaded = st.session_state.master_df is not None
    wells_loaded = "df_all" in st.session_state
    master_count = (
        st.session_state.master_df["Name"].nunique()
        if master_loaded and "Name" in st.session_state.master_df.columns else 0
    )
    interpretation_count = (
        st.session_state.df_all["Well"].nunique()
        if wells_loaded and "Well" in st.session_state.df_all.columns else 0
    )

    st.markdown("### Workspace Status")
    status_cols = st.columns(3)

    with status_cols[0]:
        dot_class = "rp-status-dot" if master_loaded else "rp-status-dot rp-status-dot-muted"
        text = f"{master_count} wells loaded" if master_loaded else "Waiting for master table"
        st.markdown(
            f"""
            <div class="rp-card">
                <div class="rp-chip">Well Map</div>
                <div class="rp-status"><span class="{dot_class}"></span>{text}</div>
                <p>Upload well coordinates and symbols to build the field location map.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with status_cols[1]:
        dot_class = "rp-status-dot" if wells_loaded else "rp-status-dot rp-status-dot-muted"
        text = f"{interpretation_count} wells integrated" if wells_loaded else "Waiting for interpretation files"
        st.markdown(
            f"""
            <div class="rp-card">
                <div class="rp-chip">Multi-Well</div>
                <div class="rp-status"><span class="{dot_class}"></span>{text}</div>
                <p>Merge interpretation spreadsheets, select zones, and compare reservoir properties.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with status_cols[2]:
        st.markdown(
            """
            <div class="rp-card-dark">
                <div class="rp-chip">Pore Typing</div>
                <h3>Carbonate pore throat analysis</h3>
                <p>Run supervised pore typing or autonomous capillary-curve classification workflows.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Start a Workflow")
    module_cols = st.columns(4)
    modules = [
        ("Well Map", "Load master well table and inspect well locations."),
        ("Multi-Well Analysis", "Merge interpretation results and compare zones."),
        ("Pore Typing", "Use trained pore typing methods and RCA overlays."),
        ("Autonomous Pore Typing", "Classify carbonate MICP curves without labels."),
    ]

    for col, (target, description) in zip(module_cols, modules):
        with col:
            st.markdown(
                f"""
                <div class="rp-step">
                    <strong>{target}</strong>
                    <span>{description}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.button(
                f"Open {target}",
                key=f"home_open_{target}",
                on_click=_set_navigation,
                args=(target,)
            )

    st.markdown("### Recommended Analysis Path")
    step_cols = st.columns(4)
    steps = [
        ("1. Load well metadata", "Start in Well Map with Name, symbol, surface coordinates, and target."),
        ("2. Integrate interpretation files", "Upload per-well interpretation spreadsheets and select target zones."),
        ("3. Classify pore systems", "Use Pore Typing or Autonomous Pore Typing depending on label availability."),
        ("4. Review RCA and exports", "Inspect FZI-constrained RCA, capillary curves, and classification results."),
    ]
    for col, (title, body) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="rp-step">
                    <strong>{title}</strong>
                    <span>{body}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    if master_loaded:
        st.markdown("### Well Overview")
        st.dataframe(st.session_state.master_df.head(8), use_container_width=True)

    st.markdown("### Local Data Notice")
    st.info(
        "Large reservoir datasets, split workbooks, model binaries, logs, and outputs are kept local "
        "and ignored by Git. Put source datasets in `data/`, regenerate split files when needed, "
        "and avoid uploading sensitive well data to GitHub."
    )

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


