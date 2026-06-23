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
if "filtered_perforation_df" not in st.session_state:
    st.session_state.filtered_perforation_df = pd.DataFrame()


def _set_navigation(target):
    st.session_state.navigation = target


def _weighted_avg(group, col):
    thickness = group["TVDSS_THK"].sum()
    if thickness == 0:
        return 0
    return (group[col] * group["TVDSS_THK"]).sum() / thickness


def _show_foldable_table(title, df, expanded=False):
    with st.expander(title, expanded=expanded):
        st.dataframe(df, use_container_width=True)


PERFORATION_REQUIRED_COLUMNS = [
    "Well Name",
    "Perforation Date",
    "Perforation Formation_old",
    "Perforation Formation_updated",
    "Perforation Top (MD, m)",
    "Perforation Base (MD, m)",
    "射孔井段 (m)",
]


def _load_perforation_table(file):
    df = pd.read_excel(file)
    missing = [col for col in PERFORATION_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required perforation columns: {missing}")

    df = df.copy()
    df["Well Name"] = df["Well Name"].astype(str).str.strip()
    df["Perforation Date"] = pd.to_datetime(df["Perforation Date"], errors="coerce")
    numeric_cols = ["Perforation Top (MD, m)", "Perforation Base (MD, m)", "射孔井段 (m)"]
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
        top_md = group["Perforation Top (MD, m)"].min()
        base_md = group["Perforation Base (MD, m)"].max()
        md_range = ""
        if pd.notna(top_md) and pd.notna(base_md):
            md_range = f"{top_md:.1f} - {base_md:.1f}"

        rows.append({
            "Name": str(well_name).strip(),
            "Perforation Count": len(group),
            "Perforation Formations": ", ".join(formations),
            "Perforation Date Range": _format_date_range(group["Perforation Date"]),
            "Perforation MD Range": md_range,
            "Total Perforated Interval (m)": group["射孔井段 (m)"].sum(),
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


def _render_perforation_formation_filter(perforation_df):
    if perforation_df is None or perforation_df.empty:
        st.session_state.filtered_perforation_df = pd.DataFrame()
        return pd.DataFrame()

    well_col = "Well Name"
    if well_col not in perforation_df.columns:
        st.session_state.filtered_perforation_df = pd.DataFrame()
        return pd.DataFrame()

    with st.expander("Table B filter by Well Name", expanded=False):
        well_values = sorted(
            perforation_df[well_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        selected_wells = st.multiselect(
            "Select Well Name",
            options=well_values,
            default=well_values,
            key="perforation_well_name_filter"
        )
        if selected_wells:
            filtered = perforation_df[
                perforation_df[well_col].astype(str).isin(selected_wells)
            ].copy()
        else:
            filtered = perforation_df.copy()
            st.warning("No well selected. Showing all table B rows.")

        st.caption(f"{len(filtered):,} of {len(perforation_df):,} perforation row(s) shown.")
        st.dataframe(filtered, use_container_width=True)
        st.session_state.filtered_perforation_df = filtered
        return filtered


def _filter_interpretation_by_perforation_depth(df_all, perforation_df):
    interp_required = ["Well", "MD_TOP", "MD_BOTTOM"]
    perf_required = ["Well Name", "Perforation Top (MD, m)", "Perforation Base (MD, m)"]
    missing_interp = [col for col in interp_required if col not in df_all.columns]
    missing_perf = [col for col in perf_required if col not in perforation_df.columns]

    if missing_interp or missing_perf:
        if missing_interp:
            st.warning(f"Depth filter unavailable: interpretation files are missing {missing_interp}.")
        if missing_perf:
            st.warning(f"Depth filter unavailable: Table B is missing {missing_perf}.")
        return df_all.iloc[0:0].copy(), pd.DataFrame()

    df_work = df_all.copy()
    df_work["Well"] = df_work["Well"].astype(str).str.strip()
    df_work["MD_TOP"] = pd.to_numeric(df_work["MD_TOP"], errors="coerce")
    df_work["MD_BOTTOM"] = pd.to_numeric(df_work["MD_BOTTOM"], errors="coerce")
    df_work["__InterpretationRowId"] = df_work.index

    perf_work = perforation_df.copy()
    perf_work["Well Name"] = perf_work["Well Name"].astype(str).str.strip()
    perf_work["Perforation Top (MD, m)"] = pd.to_numeric(
        perf_work["Perforation Top (MD, m)"],
        errors="coerce"
    )
    perf_work["Perforation Base (MD, m)"] = pd.to_numeric(
        perf_work["Perforation Base (MD, m)"],
        errors="coerce"
    )
    perf_work = perf_work.dropna(subset=["Well Name", "Perforation Top (MD, m)", "Perforation Base (MD, m)"])

    filtered_parts = []
    summary_rows = []
    for perforation_idx, perf_row in perf_work.iterrows():
        well_name = str(perf_row["Well Name"]).strip()
        depth_top = perf_row["Perforation Top (MD, m)"]
        depth_base = perf_row["Perforation Base (MD, m)"]
        well_rows = df_work[df_work["Well"] == str(well_name).strip()].copy()

        if pd.isna(depth_top) or pd.isna(depth_base):
            selected = well_rows.iloc[0:0].copy()
        else:
            selected = well_rows[
                (well_rows["MD_TOP"] <= depth_base)
                & (well_rows["MD_BOTTOM"] >= depth_top)
            ].copy()

        if not selected.empty:
            filtered_parts.append(selected)

        summary_rows.append({
            "Well": str(well_name).strip(),
            "Perforation Top (MD, m)": depth_top,
            "Perforation Base (MD, m)": depth_base,
            "Row-level condition": "MD_TOP <= perforation base and MD_BOTTOM >= perforation top",
            "Matched interpretation rows": len(selected),
        })

    if filtered_parts:
        filtered = (
            pd.concat(filtered_parts, ignore_index=True)
            .drop_duplicates(subset=["__InterpretationRowId"])
            .drop(columns=["__InterpretationRowId"])
            .reset_index(drop=True)
        )
    else:
        filtered = df_work.iloc[0:0].drop(columns=["__InterpretationRowId"]).copy()

    return filtered, pd.DataFrame(summary_rows)


def _render_perforation_row_level_charts(df_filtered):
    if df_filtered.empty:
        st.warning("No interpretation rows remain after filtering.")
        return

    required_cols = ["Well", "ZONE", "MD_TOP", "MD_BOTTOM", "TVDSS_THK", "PHIE", "VSH", "SWE"]
    missing_cols = [col for col in required_cols if col not in df_filtered.columns]
    if missing_cols:
        st.info(f"Row-level plots not available: missing {missing_cols}.")
        return

    df_plot = df_filtered.copy().reset_index(drop=True)
    df_plot["Layer No."] = df_plot.index + 1
    df_plot["Layer"] = df_plot.apply(
        lambda row: (
            f"{int(row['Layer No.'])}. {row['Well']} | {row['ZONE']} | "
            f"{row['MD_TOP']:.1f}-{row['MD_BOTTOM']:.1f}"
        ),
        axis=1
    )

    st.markdown("### Thickness Comparison")
    fig_thk = px.bar(
        df_plot,
        x="Layer",
        y="TVDSS_THK",
        color="ZONE",
        title="Thickness by Selected Interpretation Row",
        hover_data=["Layer No.", "Well", "ZONE", "MD_TOP", "MD_BOTTOM"]
    )
    fig_thk.update_layout(xaxis_title="Selected interpretation row")
    st.plotly_chart(fig_thk, use_container_width=True)

    st.markdown("### Reservoir Properties")
    for property_name in ["PHIE", "VSH", "SWE"]:
        fig_prop = px.bar(
            df_plot,
            x="Layer",
            y=property_name,
            color="ZONE",
            title=f"{property_name} by Selected Interpretation Row",
            hover_data=["Layer No.", "Well", "ZONE", "MD_TOP", "MD_BOTTOM"]
        )
        fig_prop.update_layout(xaxis_title="Selected interpretation row")
        st.plotly_chart(fig_prop, use_container_width=True)


def _render_multi_well_analysis(df_all, perforation_filter_df=None):
    st.markdown("### ZONE Selection")

    has_perforation_filter = perforation_filter_df is not None and not perforation_filter_df.empty
    use_perforation_depth = False
    if has_perforation_filter:
        use_perforation_depth = st.checkbox(
            "Use Table B well-name filter and perforation depth range",
            value=False,
            key="use_perforation_depth_filter"
        )

    if use_perforation_depth:
        st.info(
            "ZONE selection is disabled. Each interpretation row is kept when its MD_TOP and "
            "MD_BOTTOM overlap an individual selected Table B perforation interval: "
            "MD_TOP <= Perforation Base and MD_BOTTOM >= Perforation Top."
        )
        df_filtered, depth_summary = _filter_interpretation_by_perforation_depth(
            df_all,
            perforation_filter_df
        )
        if not depth_summary.empty:
            _show_foldable_table("Perforation Depth Filter Summary", depth_summary, expanded=False)
    else:
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

    if use_perforation_depth:
        _render_perforation_row_level_charts(df_filtered)
        return

    st.markdown("### Combined Data")
    _show_foldable_table("Combined Data Table", df_filtered, expanded=False)

    if df_filtered.empty:
        st.warning("No interpretation rows remain after filtering.")
        return

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
                _show_foldable_table("Summary Table", summary_df, expanded=False)

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
    - Perforation Top (MD, m)
    - Perforation Base (MD, m)
    - 射孔井段 (m)
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

    if st.session_state.master_df is not None:
        master_df = st.session_state.master_df.copy()
        master_df["Name"] = master_df["Name"].astype(str).str.strip()
        map_df = _merge_master_with_perforation_summary(
            master_df,
            st.session_state.perforation_df
        )

        st.markdown("### Well Map")
        st.plotly_chart(plot_well_map(map_df), use_container_width=True)

        detail_df = _merge_master_with_perforation_detail(
            master_df,
            st.session_state.perforation_df
        )
        if not detail_df.empty:
            with st.expander("A + B detailed perforation table", expanded=False):
                st.dataframe(detail_df, use_container_width=True)
    else:
        st.info("No master table loaded. Well map is skipped, and Multi-Well Analysis will use uploaded interpretation files directly.")

    if st.session_state.perforation_df is not None:
        _render_perforation_formation_filter(st.session_state.perforation_df)
    else:
        st.session_state.filtered_perforation_df = pd.DataFrame()

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

    uploaded_wells = sorted(df_all["Well"].dropna().unique())
    st.markdown("### Well Selection From Interpretation Files")
    selected_wells = st.multiselect(
        "Select wells to analyze",
        options=uploaded_wells,
        default=uploaded_wells,
        key="interpretation_well_selector"
    )

    df_selected_wells = df_all[df_all["Well"].isin(selected_wells)].copy()

    if df_selected_wells.empty:
        st.warning("No uploaded interpretation data matches the selected wells.")
        return

    _render_multi_well_analysis(
        df_selected_wells,
        st.session_state.get("filtered_perforation_df")
    )


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
        _show_foldable_table("Well Overview Table", st.session_state.master_df.head(8), expanded=False)

    st.markdown("### Local Data Notice")
    st.info(
        "Large reservoir datasets, split workbooks, model binaries, logs, and outputs are kept local "
        "and ignored by Git. Put source datasets in `data/`, regenerate split files when needed, "
        "and avoid uploading sensitive well data to GitHub."
    )

elif menu == "Well Analysis":
    _render_well_analysis_workspace()
