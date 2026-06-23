import streamlit as st
import pandas as pd
import plotly.express as px

from modules.uploader import load_master_table
from modules.mapping import plot_well_map
from modules.well_data import load_and_merge_well_data
from modules import pore_typing_ui
from modules import autonomous_pore_typing_ui
from modules.ui_theme import apply_theme


st.set_page_config(layout="wide")
apply_theme()

st.title("Reservoir Analysis Platform")

if "navigation" not in st.session_state:
    st.session_state.navigation = "Home"
if "master_df" not in st.session_state:
    st.session_state.master_df = None
if "perforation_df" not in st.session_state:
    st.session_state.perforation_df = None


def _set_navigation(target):
    st.session_state.navigation = target


def _weighted_avg(group, col):
    thickness = group["TVDSS_THK"].sum()
    if thickness == 0:
        return 0
    return (group[col] * group["TVDSS_THK"]).sum() / thickness


PERFORATION_REQUIRED_COLUMNS = [
    "Well Name",
    "Perforation Date",
    "Perforation Formation_old",
    "Perforation Formation_updated",
    "Perforation Top （MD，m）",
    "Perforation Base （MD，m）",
    "射孔井段（m）",
]


def _load_perforation_table(file):
    df = pd.read_excel(file)
    missing = [col for col in PERFORATION_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required perforation columns: {missing}")

    df = df.copy()
    df["Well Name"] = df["Well Name"].astype(str).str.strip()
    df["Perforation Date"] = pd.to_datetime(df["Perforation Date"], errors="coerce")
    numeric_cols = ["Perforation Top （MD，m）", "Perforation Base （MD，m）", "射孔井段（m）"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _format_date_range(values):
    values = values.dropna()
    if values.empty:
        return ""
    min_date = values.min().date().isoformat()
    max_date = values.max().date().isoformat()
    return min_date if min_date == max_date else f"{min_date} to {max_date}"


def _summarize_perforations(perforation_df):
    if perforation_df is None or perforation_df.empty:
        return pd.DataFrame()

    rows = []
    for well_name, group in perforation_df.groupby("Well Name", dropna=False):
        formations = (
            group["Perforation Formation_updated"]
            .fillna(group["Perforation Formation_old"])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        top_md = group["Perforation Top （MD，m）"].min()
        base_md = group["Perforation Base （MD，m）"].max()
        md_range = ""
        if pd.notna(top_md) and pd.notna(base_md):
            md_range = f"{top_md:.1f} - {base_md:.1f}"

        rows.append({
            "Name": str(well_name).strip(),
            "Perforation Count": len(group),
            "Perforation Formations": ", ".join(formations),
            "Perforation Date Range": _format_date_range(group["Perforation Date"]),
            "Perforation MD Range": md_range,
            "Total Perforated Interval (m)": group["射孔井段（m）"].sum(),
        })

    return pd.DataFrame(rows)


def _merge_master_with_perforation_summary(master_df, perforation_df):
    if perforation_df is None or perforation_df.empty:
        return master_df.copy()

    perforation_summary = _summarize_perforations(perforation_df)
    if perforation_summary.empty:
        return master_df.copy()

    return master_df.merge(perforation_summary, on="Name", how="left")


def _merge_master_with_perforation_detail(master_df, perforation_df):
    if perforation_df is None or perforation_df.empty:
        return pd.DataFrame()

    detail = perforation_df.rename(columns={"Well Name": "Name"}).copy()
    return master_df.merge(detail, on="Name", how="left")


def _render_multi_well_analysis(df_all):
    st.markdown("### ZONE Selection")

    zones = sorted(df_all["ZONE"].dropna().unique())
    selected_zones = st.multiselect(
        "Select ZONE (optional)",
        options=zones,
        default=[],
        key="zone_selector"
    )

    if selected_zones:
        df_filtered = df_all[df_all["ZONE"].isin(selected_zones)].copy()
    else:
        df_filtered = df_all.copy()

    numeric_cols = ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"]
    for col in numeric_cols:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    st.markdown("### Combined Data")
    st.dataframe(df_filtered, use_container_width=True)

    st.markdown("### Summary")
    required_cols = ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"]
    if not all(col in df_filtered.columns for col in required_cols):
        st.info("Summary not available: missing required columns.")
    else:
        df_calc = df_filtered.dropna(subset=required_cols)
        if df_calc.empty:
            st.info("Summary not available: no valid data after filtering.")
        else:
            md_total = df_calc["MD_THK"].sum()
            tvd_total = df_calc["TVDSS_THK"].sum()
            if tvd_total <= 0:
                st.info("Summary not available: total thickness is zero.")
            else:
                summary_df = pd.DataFrame({
                    "Metric": ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"],
                    "Value": [
                        md_total,
                        tvd_total,
                        (df_calc["VSH"] * df_calc["TVDSS_THK"]).sum() / tvd_total,
                        (df_calc["PHIE"] * df_calc["TVDSS_THK"]).sum() / tvd_total,
                        (df_calc["SWE"] * df_calc["TVDSS_THK"]).sum() / tvd_total,
                    ],
                })
                st.dataframe(summary_df, use_container_width=True)

    if not all(col in df_filtered.columns for col in ["Well", "ZONE", "TVDSS_THK"]):
        st.info("Thickness plot not available: missing Well, ZONE, or TVDSS_THK.")
        return

    st.markdown("### Thickness Comparison (ZONE + Total)")
    df_zone = df_filtered.groupby(["Well", "ZONE"], as_index=False)["TVDSS_THK"].sum()
    df_total = df_zone.groupby("Well", as_index=False)["TVDSS_THK"].sum()
    df_total["ZONE"] = "Total"
    df_plot = pd.concat([df_zone, df_total], ignore_index=True)

    fig_thk = px.bar(
        df_plot,
        x="Well",
        y="TVDSS_THK",
        color="ZONE",
        barmode="group",
        title="Thickness by Well and ZONE"
    )
    st.plotly_chart(fig_thk, use_container_width=True)

    st.markdown("### Reservoir Properties (ZONE + Total)")
    if not all(col in df_filtered.columns for col in ["PHIE", "VSH", "SWE"]):
        st.info("Reservoir property plot not available: missing PHIE, VSH, or SWE.")
        return

    df_prop_zone = df_filtered.groupby(["Well", "ZONE"]).apply(
        lambda g: pd.Series({
            "PHIE": _weighted_avg(g, "PHIE"),
            "VSH": _weighted_avg(g, "VSH"),
            "SWE": _weighted_avg(g, "SWE"),
        })
    ).reset_index()

    df_prop_total = df_filtered.groupby("Well").apply(
        lambda g: pd.Series({
            "PHIE": _weighted_avg(g, "PHIE"),
            "VSH": _weighted_avg(g, "VSH"),
            "SWE": _weighted_avg(g, "SWE"),
        })
    ).reset_index()
    df_prop_total["ZONE"] = "Total"

    df_prop_all = pd.concat([df_prop_zone, df_prop_total], ignore_index=True)

    for property_name in ["PHIE", "VSH", "SWE"]:
        fig_prop = px.bar(
            df_prop_all,
            x="Well",
            y=property_name,
            color="ZONE",
            barmode="group",
            title=f"{property_name} by Well, ZONE and Total"
        )
        st.plotly_chart(fig_prop, use_container_width=True)


def _render_well_analysis_workspace():
    st.subheader("Well Analysis")
    st.info(
        "Upload table A when you want a location map and master-well filtering. "
        "You can also skip table A and run Multi-Well Analysis directly from interpretation files."
    )

    st.markdown("""
    ### Master Table Requirements

    Upload table A first. It must contain:

    - Name: Well name
    - Well symbol: Oil or Injection
    - Surface X: X coordinate
    - Surface Y: Y coordinate
    - Target: Reservoir target zone
    """)

    file = st.file_uploader("Upload Master Table", type=["xlsx"], key="master_table_uploader")
    if file is not None:
        try:
            df = load_master_table(file)
            st.session_state.master_df = df
            st.success("Master table loaded")
        except Exception as exc:
            st.error(str(exc))

    st.markdown("### Perforation Table B Requirements")
    st.markdown("""
    Optional table B records perforation intervals and must contain:

    - Well Name
    - Perforation Date
    - Perforation Formation_old
    - Perforation Formation_updated
    - Perforation Top （MD，m）
    - Perforation Base （MD，m）
    - 射孔井段（m）
    """)

    perforation_file = st.file_uploader(
        "Upload Perforation Table B",
        type=["xlsx"],
        key="perforation_table_uploader"
    )
    if perforation_file is not None:
        try:
            perforation_df = _load_perforation_table(perforation_file)
            st.session_state.perforation_df = perforation_df
            st.success(
                f"Perforation table loaded: {perforation_df['Well Name'].nunique()} well(s), "
                f"{len(perforation_df)} interval row(s)"
            )
        except Exception as exc:
            st.error(str(exc))

    selected_wells = None
    if st.session_state.master_df is not None:
        master_df = st.session_state.master_df.copy()
        master_df["Name"] = master_df["Name"].astype(str).str.strip()
        map_df = _merge_master_with_perforation_summary(
            master_df,
            st.session_state.perforation_df
        )

        st.markdown("### Well Map")
        st.dataframe(map_df, use_container_width=True)
        st.plotly_chart(plot_well_map(map_df), use_container_width=True)

        detail_df = _merge_master_with_perforation_detail(
            master_df,
            st.session_state.perforation_df
        )
        if not detail_df.empty:
            with st.expander("A + B detailed perforation table", expanded=False):
                st.dataframe(detail_df, use_container_width=True)

        master_wells = sorted(master_df["Name"].dropna().astype(str).unique())
        st.markdown("### Well Selection From Table A")
        selected_wells = st.multiselect(
            "Select wells to display and analyze",
            options=master_wells,
            default=master_wells,
            key="master_well_selector"
        )

        if not selected_wells:
            st.warning("No wells selected. Select at least one well from table A.")
            return

        selected_master_df = map_df[map_df["Name"].isin(selected_wells)].copy()
        st.caption(f"{len(selected_wells)} of {len(master_wells)} wells selected from table A.")
        st.plotly_chart(plot_well_map(selected_master_df), use_container_width=True)
    else:
        st.info("No master table loaded. Well map is skipped, and Multi-Well Analysis will use uploaded interpretation files directly.")

    st.markdown("### Upload Well Interpretation Files")
    st.caption(
        "Upload one or more interpretation Excel files. If a file does not contain a `Well` column, "
        "the file name without extension is used as the well name."
    )

    files = st.file_uploader(
        "Upload Well Interpretation Files",
        type=["xlsx"],
        accept_multiple_files=True,
        key="well_uploader"
    )

    if files:
        try:
            df_all = load_and_merge_well_data(files)
            df_all["Well"] = df_all["Well"].astype(str).str.strip()
            st.session_state.df_all = df_all
            st.success(f"Loaded {df_all['Well'].nunique()} interpretation well(s)")
        except Exception as exc:
            st.error(str(exc))

    if "df_all" not in st.session_state:
        st.warning("Please upload well interpretation files to continue multi-well analysis.")
        return

    df_all = st.session_state.df_all.copy()
    df_all["Well"] = df_all["Well"].astype(str).str.strip()

    if selected_wells is None:
        uploaded_wells = sorted(df_all["Well"].dropna().unique())
        st.markdown("### Well Selection From Interpretation Files")
        selected_wells = st.multiselect(
            "Select wells to analyze",
            options=uploaded_wells,
            default=uploaded_wells,
            key="interpretation_well_selector"
        )

    df_selected_wells = df_all[df_all["Well"].isin(selected_wells)].copy()

    if st.session_state.master_df is not None:
        missing_interpretation = sorted(set(selected_wells) - set(df_selected_wells["Well"].unique()))
        uploaded_not_selected = sorted(set(df_all["Well"].unique()) - set(selected_wells))

        if missing_interpretation:
            st.warning(
                "No interpretation data matched these selected master wells: "
                + ", ".join(missing_interpretation)
            )
        if uploaded_not_selected:
            st.caption(
                "Uploaded interpretation wells hidden by table A selection: "
                + ", ".join(uploaded_not_selected)
            )

    if df_selected_wells.empty:
        st.warning("No uploaded interpretation data matches the selected wells.")
        return

    _render_multi_well_analysis(df_selected_wells)


menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Well Analysis",
        "Pore Typing",
        "Autonomous Pore Typing"
    ],
    key="navigation"
)

if menu == "Pore Typing":
    pore_typing_ui.run()

elif menu == "Autonomous Pore Typing":
    autonomous_pore_typing_ui.run()

elif menu == "Home":
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
                <div class="rp-chip">Well Analysis</div>
                <div class="rp-status"><span class="{dot_class}"></span>{text}</div>
                <p>Upload well metadata to build maps, or run interpretation analysis directly.</p>
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
                <div class="rp-chip">Interpretation</div>
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
    modules = [
        ("Well Analysis", "Map wells when metadata is available and compare multi-well interpretation results."),
        ("Pore Typing", "Use trained pore typing methods and RCA overlays."),
        ("Autonomous Pore Typing", "Classify carbonate MICP curves without labels."),
    ]
    module_cols = st.columns(len(modules))

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
        ("1. Open Well Analysis", "Upload table A for mapping, or skip directly to interpretation files."),
        ("2. Select wells", "Use table A well names when available, otherwise select wells from uploaded interpretation files."),
        ("3. Integrate interpretation files", "Upload per-well interpretation spreadsheets named by well or containing a Well column."),
        ("4. Review reservoir and pore outputs", "Compare zones, then continue to pore typing workflows when needed."),
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

elif menu == "Well Analysis":
    _render_well_analysis_workspace()
