import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from modules.ui_theme import COLORS, TYPE_COLORS, style_plotly


REQUIRED_COLUMNS = ["CPOR_clean", "CKH_clean", "PC_STRESS_CORR", "SW_STRESS_CORR"]
OPTIONAL_COLUMNS = ["PTR_P", "PORE_V_P"]
FORMATION_COLUMN_CANDIDATES = [
    "Formation", "formation", "FORMATION",
    "Layer", "layer", "LAYER",
    "Zone", "zone", "ZONE",
    "Member", "member", "MEMBER",
    "Stratigraphy", "stratigraphy",
    "Strata", "strata",
    "Reservoir", "reservoir",
]
SAMPLE_ID_CANDIDATES = [
    "SampleID", "Sample_ID", "Sample", "Plug", "Plug_ID", "Core_ID",
    "ID", "ReferenceName", "wellName", "Well", "WELL", "WellName_2",
    "Core", "CoreNo"
]
SAMPLE_ID_COMBINATIONS = [
    ("WellName_2", "Plug No"),
    ("WellName_2", "ReferenceName"),
    ("WellName_2", "ID"),
    ("wellName", "Plug No"),
    ("wellName", "ReferenceName"),
    ("Well", "Plug"),
    ("WELL", "Plug"),
]

COLOR_MAP = TYPE_COLORS

SCATTER_MARKER_SIZE = 4
FEATURE_MARKER_SIZE = 6
CURVE_INTERPOLATION_POINTS = 160
CARBONATE_PC_SATURATIONS = [0.10, 0.25, 0.50, 0.75, 0.90]
PORE_RADIUS_BINS = 40
AUTO_CLUSTER_MIN = 2
AUTO_CLUSTER_MAX = 8
WASHBURN_IFT_DYNE_PER_CM = 480
WASHBURN_CONTACT_ANGLE_DEG = 140
PORE_RADIUS_MIN_UM = 0.001
PORE_RADIUS_MAX_UM = 1000
PORE_RADIUS_CLASSES_PER_DECADE = 15


def _style_figure(fig, legend_title=None):
    return style_plotly(fig, legend_title=legend_title)


def _valid_curve_group_count(df, sample_col):
    grouped = df.dropna(subset=[sample_col]).groupby(sample_col)
    group_sizes = grouped.size()
    unique_sw = grouped["SW_STRESS_CORR"].nunique()
    valid_groups = (group_sizes >= 2) & (unique_sw >= 2)
    return int(valid_groups.sum()), int(group_sizes.size)


def _find_sample_col(df):
    ranked_candidates = []

    for col in SAMPLE_ID_CANDIDATES:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if values.empty:
            continue

        valid_curve_groups, _ = _valid_curve_group_count(df, col)
        unique_count = int(values.nunique())

        if unique_count > 1 and valid_curve_groups >= 2:
            ranked_candidates.append((valid_curve_groups, unique_count, col))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(reverse=True)
    return ranked_candidates[0][2]


def _auto_assign_sample_identifier(df):
    df = df.copy()
    ranked_ids = []

    for col in SAMPLE_ID_CANDIDATES:
        if col not in df.columns or df[col].dropna().empty:
            continue

        valid_count, total_count = _valid_curve_group_count(df, col)
        unique_count = int(df[col].dropna().nunique())
        if unique_count > 1 and valid_count >= 2:
            ranked_ids.append({
                "valid_count": valid_count,
                "total_count": total_count,
                "unique_count": unique_count,
                "sample_col": col,
                "df": df,
                "message": (
                    f"Using `{col}` as the sample identifier "
                    f"({valid_count} of {total_count} groups have valid capillary curves)."
                ),
            })

    for cols in SAMPLE_ID_COMBINATIONS:
        if not set(cols).issubset(df.columns):
            continue

        auto_col = "Core_ID"
        values = [
            df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
            for col in cols
        ]
        auto_values = values[0]
        for value in values[1:]:
            auto_values = auto_values + "_" + value

        candidate = df.copy()
        candidate[auto_col] = auto_values
        valid_count, total_count = _valid_curve_group_count(candidate, auto_col)
        if valid_count >= 2:
            unique_count = int(candidate[auto_col].dropna().nunique())
            ranked_ids.append({
                "valid_count": valid_count,
                "total_count": total_count,
                "unique_count": unique_count,
                "sample_col": auto_col,
                "df": candidate,
                "message": (
                    "No existing sample identifier was better than the generated one, so "
                    f"`{auto_col}` was generated from {' + '.join(cols)} "
                    f"({valid_count} of {total_count} groups have valid capillary curves)."
                ),
            })

    if ranked_ids:
        ranked_ids.sort(
            reverse=True,
            key=lambda item: (item["valid_count"], item["unique_count"]),
        )
        selected = ranked_ids[0]
        return selected["df"], selected["sample_col"], selected["message"]

    raise ValueError(
        "No valid core/plug sample identifier could be found or generated. "
        "Provide a column such as `Core_ID`, `SampleID`, `Plug_ID`, `ID`, or "
        "`ReferenceName`, or provide columns such as `WellName_2` + `Plug No` so "
        "the app can build `Core_ID`. Each core/plug must have at least two valid "
        "`SW_STRESS_CORR` and `PC_STRESS_CORR` points."
    )


def _clean_input(df):
    df = df.copy()

    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[
        (df["CPOR_clean"] > 0) &
        (df["CPOR_clean"] < 1) &
        (df["CKH_clean"] > 0) &
        (df["PC_STRESS_CORR"] > 0) &
        (df["SW_STRESS_CORR"] >= 0) &
        (df["SW_STRESS_CORR"] <= 1)
    ].copy()

    if "PTR_P" in df.columns:
        df = df[(df["PTR_P"].isna()) | (df["PTR_P"] > 0)].copy()

    return df


def _formation_filter_candidates(df):
    candidates = [
        col for col in FORMATION_COLUMN_CANDIDATES
        if col in df.columns and df[col].dropna().nunique() > 1
    ]
    if candidates:
        return candidates

    fallback_candidates = []
    for col in df.columns:
        if col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        unique_count = values.nunique()
        if 1 < unique_count <= min(80, max(2, len(values) // 2)):
            fallback_candidates.append(col)

    return fallback_candidates


def _render_formation_filter(df):
    candidates = _formation_filter_candidates(df)
    if not candidates:
        st.info(
            "No formation/layer column was detected, so all uploaded data will be analyzed together."
        )
        return df, None, []

    st.markdown("### Data filter before autonomous typing")
    filter_cols = st.columns([1, 2])
    with filter_cols[0]:
        formation_col = st.selectbox(
            "Formation / layer column",
            options=candidates,
            key="autonomous_formation_filter_column"
        )

    formation_values = (
        df[formation_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    with filter_cols[1]:
        selected_formations = st.multiselect(
            "Formations to analyze",
            options=formation_values,
            default=formation_values,
            key=f"autonomous_formation_filter_values_{formation_col}"
        )

    if not selected_formations:
        st.warning("No formation selected. Showing all data instead.")
        selected_formations = formation_values

    filtered = df[df[formation_col].astype(str).isin(selected_formations)].copy()
    st.caption(
        f"Using {len(filtered):,} of {len(df):,} cleaned rows from "
        f"{len(selected_formations)} selected formation(s)."
    )
    return filtered, formation_col, selected_formations


def _add_petrophysical_features(df):
    df = df.copy()

    phi = df["CPOR_clean"]
    k = df["CKH_clean"]

    df["logK"] = np.log10(k)
    df["logPc"] = np.log10(df["PC_STRESS_CORR"])
    df["FZI"] = 0.0314 * np.sqrt(k / phi) * (1 - phi) / phi
    df["logFZI"] = np.log10(df["FZI"])
    df["R35_equiv"] = k * (1 - phi) / phi
    df["logR35_equiv"] = np.log10(df["R35_equiv"])

    if "PTR_P" in df.columns:
        df["logPTR"] = np.where(df["PTR_P"] > 0, np.log10(df["PTR_P"]), np.nan)

    return df


def _carbonate_capillary_shape_metrics(group):
    curve = (
        group[["SW_STRESS_CORR", "logPc"]]
        .dropna()
        .groupby("SW_STRESS_CORR", as_index=False)["logPc"]
        .median()
        .sort_values("SW_STRESS_CORR")
    )

    metric_names = [
        "Pc_log_at_sw_10",
        "Pc_log_at_sw_25",
        "Pc_log_at_sw_50",
        "Pc_log_at_sw_75",
        "Pc_log_at_sw_90",
        "Pc_log_span",
        "Pc_log_entry_to_tail",
        "Pc_slope_entry",
        "Pc_slope_middle",
        "Pc_slope_tail",
        "Pc_slope_complexity",
    ]

    if len(curve) < 2:
        return pd.Series({name: np.nan for name in metric_names})

    sw = curve["SW_STRESS_CORR"].to_numpy()
    log_pc = curve["logPc"].to_numpy()
    pc_at_sw = {
        f"Pc_log_at_sw_{int(s * 100):02d}": np.interp(s, sw, log_pc)
        for s in CARBONATE_PC_SATURATIONS
    }

    slopes = np.diff(log_pc) / np.diff(sw)
    slopes = slopes[np.isfinite(slopes)]

    metrics = {
        **pc_at_sw,
        "Pc_log_span": np.nanmax(log_pc) - np.nanmin(log_pc),
        "Pc_log_entry_to_tail": pc_at_sw["Pc_log_at_sw_10"] - pc_at_sw["Pc_log_at_sw_90"],
        "Pc_slope_entry": (pc_at_sw["Pc_log_at_sw_25"] - pc_at_sw["Pc_log_at_sw_10"]) / 0.15,
        "Pc_slope_middle": (pc_at_sw["Pc_log_at_sw_75"] - pc_at_sw["Pc_log_at_sw_25"]) / 0.50,
        "Pc_slope_tail": (pc_at_sw["Pc_log_at_sw_90"] - pc_at_sw["Pc_log_at_sw_75"]) / 0.15,
        "Pc_slope_complexity": np.nanstd(slopes) if len(slopes) else 0.0,
    }

    return pd.Series(metrics)


def _build_sample_feature_table(df, sample_col):
    df = df.copy()

    if sample_col is None:
        df["__SampleKey"] = np.arange(len(df))
        sample_col = "__SampleKey"

    agg_spec = {
        "CPOR_clean": "median",
        "CKH_clean": "median",
        "logK": "median",
        "FZI": "median",
        "logFZI": "median",
        "R35_equiv": "median",
        "logR35_equiv": "median",
        "PC_STRESS_CORR": ["median", "max"],
        "logPc": ["median", "max"],
        "SW_STRESS_CORR": ["min", "median", "max"],
    }

    if "logPTR" in df.columns:
        agg_spec["logPTR"] = "median"
    if "PORE_V_P" in df.columns:
        agg_spec["PORE_V_P"] = "median"

    sample_features = df.groupby(sample_col).agg(agg_spec)
    sample_features.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        for col in sample_features.columns
    ]
    sample_features = sample_features.reset_index()

    carbonate_shape_metrics = (
        df.groupby(sample_col)
        .apply(_carbonate_capillary_shape_metrics)
        .reset_index()
    )
    sample_features = sample_features.merge(carbonate_shape_metrics, on=sample_col, how="left")

    feature_cols = [
        "CPOR_clean_median",
        "logK_median",
        "logFZI_median",
        "logR35_equiv_median",
        "logPc_median",
        "logPc_max",
        "Pc_log_at_sw_10",
        "Pc_log_at_sw_25",
        "Pc_log_at_sw_50",
        "Pc_log_at_sw_75",
        "Pc_log_at_sw_90",
        "Pc_log_span",
        "Pc_log_entry_to_tail",
        "Pc_slope_entry",
        "Pc_slope_middle",
        "Pc_slope_tail",
        "Pc_slope_complexity",
        "SW_STRESS_CORR_min",
        "SW_STRESS_CORR_median",
        "SW_STRESS_CORR_max",
    ]

    if "logPTR_median" in sample_features.columns:
        feature_cols.append("logPTR_median")
    if "PORE_V_P_median" in sample_features.columns:
        feature_cols.append("PORE_V_P_median")

    total_sample_count = len(sample_features)
    sample_features = sample_features.dropna(subset=feature_cols).copy()
    if len(sample_features) < 2:
        raise ValueError(
            "At least two valid samples are required for autonomous classification. "
            f"The selected sample identifier is `{sample_col}`. "
            f"{len(sample_features)} of {total_sample_count} grouped samples remain valid "
            "after carbonate capillary-curve feature extraction. Check that the sample "
            "identifier separates cores/plugs rather than only wells, and that each sample "
            "has at least two valid capillary-pressure points."
        )

    return sample_features, sample_col, feature_cols


def _choose_autonomous_cluster_count(X_scaled, random_state):
    sample_count = len(X_scaled)
    if sample_count < 2:
        raise ValueError(
            f"At least two valid samples are required for autonomous classification; "
            f"only {sample_count} valid sample is available."
        )

    if sample_count == 2:
        diagnostics = pd.DataFrame([{
            "ClusterCount": 2,
            "Silhouette": np.nan,
            "MinClusterSize": 1,
            "BalanceScore": 1.0,
            "SelectionScore": np.nan,
        }])
        return 2, diagnostics

    max_clusters = min(AUTO_CLUSTER_MAX, sample_count - 1)
    candidate_rows = []

    for candidate_count in range(AUTO_CLUSTER_MIN, max_clusters + 1):
        model = KMeans(
            n_clusters=candidate_count,
            random_state=random_state,
            n_init=20
        )
        labels = model.fit_predict(X_scaled)
        cluster_sizes = np.bincount(labels, minlength=candidate_count)
        min_cluster_size = int(cluster_sizes.min())
        expected_cluster_size = sample_count / candidate_count
        balance_score = min(min_cluster_size / expected_cluster_size, 1.0)

        if len(np.unique(labels)) < 2:
            silhouette = np.nan
            selection_score = -np.inf
        else:
            silhouette = silhouette_score(X_scaled, labels)
            selection_score = silhouette + (0.08 * balance_score) - (0.03 * (candidate_count - 2))

        candidate_rows.append({
            "ClusterCount": candidate_count,
            "Silhouette": silhouette,
            "MinClusterSize": min_cluster_size,
            "BalanceScore": balance_score,
            "SelectionScore": selection_score,
        })

    diagnostics = pd.DataFrame(candidate_rows)
    valid_diagnostics = diagnostics.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["SelectionScore"]
    )
    if valid_diagnostics.empty:
        return AUTO_CLUSTER_MIN, diagnostics

    best_row = valid_diagnostics.sort_values(
        ["SelectionScore", "Silhouette", "BalanceScore"],
        ascending=False
    ).iloc[0]
    return int(best_row["ClusterCount"]), diagnostics


def _cluster_samples(sample_features, feature_cols, random_state, n_clusters=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_features[feature_cols])

    if n_clusters is None:
        n_clusters, cluster_diagnostics = _choose_autonomous_cluster_count(
            X_scaled,
            random_state
        )
    else:
        n_clusters = int(min(n_clusters, len(sample_features)))
        if n_clusters < 2:
            raise ValueError(
                f"At least two valid samples are required for autonomous classification; "
                f"only {len(sample_features)} valid sample is available."
            )
        cluster_diagnostics = pd.DataFrame([{
            "ClusterCount": n_clusters,
            "Silhouette": np.nan,
            "MinClusterSize": np.nan,
            "BalanceScore": np.nan,
            "SelectionScore": np.nan,
        }])

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    raw_cluster = model.fit_predict(X_scaled)

    sample_features = sample_features.copy()
    sample_features["RawCluster"] = raw_cluster

    cluster_order = (
        sample_features.groupby("RawCluster")["FZI_median"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    cluster_to_type = {cluster: i + 1 for i, cluster in enumerate(cluster_order)}
    sample_features["AutoPoreType"] = sample_features["RawCluster"].map(cluster_to_type).astype(int)

    if len(feature_cols) >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        xy = pca.fit_transform(X_scaled)
        sample_features["PCA1"] = xy[:, 0]
        sample_features["PCA2"] = xy[:, 1]

    return sample_features, model, scaler, cluster_diagnostics


def _assign_types_to_rows(df, sample_features, sample_col):
    typed = df.merge(
        sample_features[[sample_col, "AutoPoreType"]],
        on=sample_col,
        how="left"
    )
    typed = typed.dropna(subset=["AutoPoreType"]).copy()
    typed["AutoPoreType"] = typed["AutoPoreType"].astype(int)
    return typed


def _subcluster_selected_type(sample_features, feature_cols, parent_type, n_subtypes, random_state):
    refined = sample_features.copy()
    refined["SecondaryPoreType"] = np.nan
    refined["RefinedPoreType"] = refined["AutoPoreType"].astype(int).astype(str)

    mask = refined["AutoPoreType"] == parent_type
    subset = refined[mask].copy()

    n_subtypes = int(min(n_subtypes, len(subset)))
    if n_subtypes < 2:
        raise ValueError(
            f"Type {int(parent_type)} has only {len(subset)} valid sample. "
            "At least two samples are required for secondary classification."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset[feature_cols])

    model = KMeans(n_clusters=n_subtypes, random_state=random_state, n_init=20)
    raw_subcluster = model.fit_predict(X_scaled)
    subset["RawSubCluster"] = raw_subcluster

    subcluster_order = (
        subset.groupby("RawSubCluster")["FZI_median"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    subcluster_to_type = {cluster: i + 1 for i, cluster in enumerate(subcluster_order)}
    subset["SecondaryPoreType"] = subset["RawSubCluster"].map(subcluster_to_type).astype(int)
    subset["RefinedPoreType"] = (
        subset["AutoPoreType"].astype(int).astype(str)
        + "."
        + subset["SecondaryPoreType"].astype(str)
    )

    refined.loc[mask, "SecondaryPoreType"] = subset["SecondaryPoreType"]
    refined.loc[mask, "RefinedPoreType"] = subset["RefinedPoreType"]
    return refined


def _assign_refined_types_to_rows(df, refined_features, sample_col):
    cols = [sample_col, "SecondaryPoreType", "RefinedPoreType", "FinalPoreType"]
    available_cols = [col for col in cols if col in refined_features.columns]
    typed = df.merge(refined_features[available_cols], on=sample_col, how="left")
    if "RefinedPoreType" not in typed.columns:
        typed["RefinedPoreType"] = typed["AutoPoreType"].astype(int).astype(str)
    if "FinalPoreType" not in typed.columns:
        typed["FinalPoreType"] = typed["RefinedPoreType"]
    return typed


def _apply_merge_rules(refined_features, merge_rules):
    merged = refined_features.copy()
    merged["FinalPoreType"] = merged["RefinedPoreType"]

    for selected_labels, new_label in merge_rules:
        if not selected_labels or not str(new_label).strip():
            continue
        merged.loc[
            merged["RefinedPoreType"].isin(selected_labels),
            "FinalPoreType"
        ] = str(new_label).strip()

    return merged


def _apply_manual_type_corrections(df, sample_col, corrections):
    corrected = df.copy()
    corrected["CorrectedPoreType"] = corrected["AutoPoreType"].astype(int).astype(str)

    if corrections:
        sample_keys = corrected[sample_col].astype(str)
        corrected_values = sample_keys.map(corrections)
        corrected.loc[corrected_values.notna(), "CorrectedPoreType"] = corrected_values.dropna()

    return corrected


def _render_manual_type_correction_controls(df_classified, sample_features, sample_col, key_prefix):
    valid_sample_keys = set(df_classified[sample_col].astype(str).dropna().unique())
    correction_key = "manual_pore_type_corrections"
    if correction_key not in st.session_state:
        st.session_state[correction_key] = {}
    st.session_state[correction_key] = {
        key: value
        for key, value in st.session_state[correction_key].items()
        if key in valid_sample_keys
    }

    sample_lookup = (
        sample_features[[sample_col, "AutoPoreType", "FZI_median", "Pc_log_at_sw_50"]]
        .copy()
        .sort_values(["AutoPoreType", sample_col])
    )
    sample_lookup["__SampleKey"] = sample_lookup[sample_col].astype(str)
    available_auto_types = [
        int(t)
        for t in sorted(df_classified["AutoPoreType"].dropna().unique())
    ]

    filter_cols = st.columns([1.2, 1.6, 1.4])
    with filter_cols[0]:
        selected_filter_types = st.multiselect(
            "Filter by auto type",
            options=available_auto_types,
            default=available_auto_types,
            format_func=lambda value: f"Type {int(value)}",
            key=f"{key_prefix}_filter_auto_types"
        )

    pc50_min = float(sample_lookup["Pc_log_at_sw_50"].min())
    pc50_max = float(sample_lookup["Pc_log_at_sw_50"].max())
    with filter_cols[1]:
        if pc50_min < pc50_max:
            pc50_range = st.slider(
                "Filter by Pc_log_at_sw_50",
                min_value=pc50_min,
                max_value=pc50_max,
                value=(pc50_min, pc50_max),
                step=max((pc50_max - pc50_min) / 100, 0.001),
                format="%.3f",
                key=f"{key_prefix}_filter_pc_log_at_sw_50"
            )
        else:
            pc50_range = (pc50_min, pc50_max)
            st.caption(
                f"Pc_log_at_sw_50 filter unavailable: all curves have "
                f"Pc_log_at_sw_50 {pc50_min:.3f}."
            )

    with filter_cols[2]:
        sample_search = st.text_input(
            "Search curve ID",
            value="",
            placeholder="Core_ID / sample name",
            key=f"{key_prefix}_filter_sample_search"
        ).strip()

    filtered_lookup = sample_lookup[
        sample_lookup["AutoPoreType"].isin(selected_filter_types)
        & sample_lookup["Pc_log_at_sw_50"].between(pc50_range[0], pc50_range[1])
    ].copy()
    if sample_search:
        filtered_lookup = filtered_lookup[
            filtered_lookup["__SampleKey"].str.contains(sample_search, case=False, na=False)
        ]

    if filtered_lookup.empty:
        st.warning("No curves match the current filters. Showing all curves instead.")
        filtered_lookup = sample_lookup.copy()

    sample_options = filtered_lookup["__SampleKey"].tolist()
    sample_labels = {
        row["__SampleKey"]: (
            f"{row[sample_col]} | Auto Type {int(row['AutoPoreType'])} | "
            f"FZI {row['FZI_median']:.2f} | Pc50 {row['Pc_log_at_sw_50']:.3f}"
        )
        for _, row in sample_lookup.iterrows()
    }
    st.caption(f"{len(sample_options)} curve(s) available after filtering.")

    correction_cols = st.columns([2, 1, 1, 1])
    with correction_cols[0]:
        selected_sample_key = st.selectbox(
            "Curve to correct",
            options=sample_options,
            format_func=lambda value: sample_labels.get(value, value),
            key=f"{key_prefix}_curve_to_correct"
        )

    current_auto_type = int(
        sample_lookup.loc[
            sample_lookup["__SampleKey"] == selected_sample_key,
            "AutoPoreType"
        ].iloc[0]
    )
    current_corrected_type = st.session_state[correction_key].get(
        selected_sample_key,
        str(current_auto_type)
    )
    available_types = [str(int(t)) for t in available_auto_types]

    with correction_cols[1]:
        correction_mode = st.radio(
            "Assignment",
            options=["Existing", "New"],
            horizontal=True,
            key=f"{key_prefix}_correction_mode"
        )

    with correction_cols[2]:
        if correction_mode == "Existing":
            new_type = st.selectbox(
                "Corrected type",
                options=available_types,
                index=available_types.index(current_corrected_type)
                if current_corrected_type in available_types else available_types.index(str(current_auto_type)),
                format_func=lambda value: f"Type {value}",
                key=f"{key_prefix}_corrected_type"
            )
        else:
            default_new_type = (
                current_corrected_type
                if current_corrected_type not in available_types
                else str(max(int(t) for t in available_types) + 1)
            )
            new_type = st.text_input(
                "New type label",
                value=default_new_type,
                key=f"{key_prefix}_new_type_label"
            ).strip()

    with correction_cols[3]:
        st.write("")
        st.write("")
        if st.button("Apply correction", key=f"{key_prefix}_apply_correction"):
            if not new_type:
                st.warning("Please enter a valid type label.")
            elif new_type == str(current_auto_type):
                st.session_state[correction_key].pop(selected_sample_key, None)
                st.rerun()
            else:
                st.session_state[correction_key][selected_sample_key] = new_type
                st.rerun()

    reset_cols = st.columns([1, 3])
    with reset_cols[0]:
        if st.button("Reset corrections", key=f"{key_prefix}_reset_corrections"):
            st.session_state[correction_key] = {}
            st.rerun()
    with reset_cols[1]:
        correction_count = len(st.session_state[correction_key])
        st.caption(f"{correction_count} manually corrected curve(s) in this session.")

    df_corrected = _apply_manual_type_corrections(
        df_classified,
        sample_col,
        st.session_state[correction_key]
    )
    return df_corrected, selected_sample_key


def _type_sort_key(label):
    parts = str(label).split(".")
    key = []
    for part in parts:
        try:
            key.append((0, int(part)))
        except ValueError:
            key.append((1, part))
    return tuple(key)


def _plot_capillary_curves(df, sample_col=None, type_col="AutoPoreType", title=None):
    fig = go.Figure()

    if type_col not in df.columns:
        return fig

    type_values = sorted(
        df[type_col].dropna().astype(str).unique(),
        key=_type_sort_key
    )

    for type_index, t in enumerate(type_values):
        group = df[df[type_col].astype(str) == t].copy()
        base_type = str(t).split(".")[0]
        try:
            base_type_number = int(base_type)
            color = COLOR_MAP.get(
                base_type_number,
                COLOR_MAP.get(((base_type_number - 1) % len(COLOR_MAP)) + 1, "gray")
            )
        except ValueError:
            color = COLOR_MAP.get((type_index % len(COLOR_MAP)) + 1, "gray")

        if sample_col in group.columns:
            sample_groups = group.groupby(sample_col, dropna=False)
        else:
            sample_groups = [(None, group)]

        for i, (sample_id, sample_group) in enumerate(sample_groups):
            sample_curve = (
                sample_group[["SW_STRESS_CORR", "PC_STRESS_CORR"]]
                .dropna()
                .groupby("SW_STRESS_CORR", as_index=False)["PC_STRESS_CORR"]
                .median()
                .sort_values("SW_STRESS_CORR")
            )
            if len(sample_curve) < 2:
                continue

            x = sample_curve["SW_STRESS_CORR"].to_numpy()
            y_log = np.log10(sample_curve["PC_STRESS_CORR"].to_numpy())
            x_smooth = np.linspace(x.min(), x.max(), CURVE_INTERPOLATION_POINTS)
            y_smooth = 10 ** np.interp(x_smooth, x, y_log)
            fzi_value = sample_group["FZI"].median() if "FZI" in sample_group.columns else np.nan
            fzi_text = f"{fzi_value:.3f}" if np.isfinite(fzi_value) else "N/A"
            shape_metrics = _carbonate_capillary_shape_metrics(sample_group)
            pc_log_at_sw_50 = shape_metrics.get("Pc_log_at_sw_50", np.nan)
            pc_slope_middle = shape_metrics.get("Pc_slope_middle", np.nan)
            pc_slope_complexity = shape_metrics.get("Pc_slope_complexity", np.nan)
            pc_entry = 10 ** np.interp(1.0, x, y_log)
            sample_text = str(sample_id) if sample_id is not None else "N/A"
            pc_log_at_sw_50_text = f"{pc_log_at_sw_50:.3f}" if np.isfinite(pc_log_at_sw_50) else "N/A"
            pc_slope_middle_text = f"{pc_slope_middle:.3f}" if np.isfinite(pc_slope_middle) else "N/A"
            pc_slope_complexity_text = f"{pc_slope_complexity:.3f}" if np.isfinite(pc_slope_complexity) else "N/A"
            pc_entry_text = f"{pc_entry:.4g}" if np.isfinite(pc_entry) else "N/A"

            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode="lines",
                    line=dict(width=1, color=color),
                    name=f"Type {t}",
                    legendgroup=f"Type {t}",
                    showlegend=(i == 0),
                    customdata=np.column_stack([
                        np.repeat(sample_text, len(x_smooth)),
                        np.repeat(str(t), len(x_smooth)),
                        np.repeat(fzi_text, len(x_smooth)),
                        np.repeat(pc_log_at_sw_50_text, len(x_smooth)),
                        np.repeat(pc_slope_middle_text, len(x_smooth)),
                        np.repeat(pc_slope_complexity_text, len(x_smooth)),
                        np.repeat(pc_entry_text, len(x_smooth)),
                    ]),
                    hovertemplate=(
                        "Sample: %{customdata[0]}<br>"
                        "Type: %{customdata[1]}<br>"
                        "FZI: %{customdata[2]}<br>"
                        "Pc_log_at_sw_50: %{customdata[3]}<br>"
                        "Pc_slope_middle: %{customdata[4]}<br>"
                        "Pc_slope_complexity: %{customdata[5]}<br>"
                        "Pc_entry at Sw=1: %{customdata[6]}<br>"
                        "Sw: %{x:.4f}<br>"
                        "Pc: %{y:.4g}<extra></extra>"
                    )
                )
            )

    fig.update_layout(
        title=title or "Mercury Injection Capillary Pressure Curves by Autonomous Pore Type",
        xaxis_title="Water Saturation / Mercury Saturation Proxy (fraction)",
        yaxis_title="Capillary Pressure (Pc)",
        yaxis=dict(type="log")
    )
    _style_figure(fig, legend_title=type_col)

    return fig


def _build_washburn_pore_throat_curve(sample_group):
    if "PC_STRESS_CORR" not in sample_group.columns or "SW_STRESS_CORR" not in sample_group.columns:
        return pd.DataFrame(), None

    micp_curve = (
        sample_group[["PC_STRESS_CORR", "SW_STRESS_CORR"]]
        .dropna()
        .query("PC_STRESS_CORR > 0 and SW_STRESS_CORR >= 0 and SW_STRESS_CORR <= 1")
        .groupby("PC_STRESS_CORR", as_index=False)["SW_STRESS_CORR"]
        .median()
        .sort_values("PC_STRESS_CORR")
    )
    if len(micp_curve) < 3:
        return pd.DataFrame(), None

    pressure = micp_curve["PC_STRESS_CORR"].to_numpy()
    saturation = micp_curve["SW_STRESS_CORR"].to_numpy()

    if np.nanmax(pressure) <= np.nanmin(pressure):
        return pd.DataFrame(), None

    if np.corrcoef(pressure, saturation)[0, 1] < 0:
        cumulative_volume = 1 - saturation
    else:
        cumulative_volume = saturation

    cumulative_volume = np.clip(cumulative_volume, 0, 1)
    cumulative_volume = np.maximum.accumulate(cumulative_volume)

    volume_span = cumulative_volume[-1] - cumulative_volume[0]
    if not np.isfinite(volume_span) or volume_span <= 0:
        return pd.DataFrame(), None

    radius_constant = (
        0.02
        * WASHBURN_IFT_DYNE_PER_CM
        * abs(np.cos(np.deg2rad(WASHBURN_CONTACT_ANGLE_DEG)))
    )
    class_count = int(
        np.log10(PORE_RADIUS_MAX_UM / PORE_RADIUS_MIN_UM)
        * PORE_RADIUS_CLASSES_PER_DECADE
    )
    radius_edges = np.logspace(
        np.log10(PORE_RADIUS_MAX_UM),
        np.log10(PORE_RADIUS_MIN_UM),
        class_count + 1
    )
    pressure_edges = radius_constant / radius_edges

    log_pressure = np.log10(pressure)
    log_pressure_edges = np.log10(pressure_edges)
    cumulative_at_edges = np.interp(
        log_pressure_edges,
        log_pressure,
        cumulative_volume,
        left=cumulative_volume[0],
        right=cumulative_volume[-1]
    )
    pore_volume = np.diff(cumulative_at_edges)
    radius_centers = np.sqrt(radius_edges[:-1] * radius_edges[1:])

    valid = np.isfinite(radius_centers) & np.isfinite(pore_volume) & (pore_volume >= 0)
    washburn_curve = pd.DataFrame({
        "PTR_P": radius_centers[valid],
        "PORE_V_P": pore_volume[valid] * 100,
    })
    positive_positions = np.flatnonzero(washburn_curve["PORE_V_P"].to_numpy() > 0)
    if len(positive_positions) < 1:
        return pd.DataFrame(), None

    start = max(int(positive_positions[0]) - 1, 0)
    stop = min(int(positive_positions[-1]) + 2, len(washburn_curve))
    washburn_curve = washburn_curve.iloc[start:stop].copy()
    washburn_curve = washburn_curve.sort_values("PTR_P")

    return washburn_curve, "Washburn from MICP; Pc bar, IFT 480 dyn/cm, theta 140 deg"


def _build_measured_pore_throat_curve(sample_group):
    if "PTR_P" not in sample_group.columns or "PORE_V_P" not in sample_group.columns:
        return pd.DataFrame(), None

    measured_curve = (
        sample_group[["PTR_P", "PORE_V_P"]]
        .dropna()
        .query("PTR_P > 0 and PORE_V_P > 0")
        .groupby("PTR_P", as_index=False)["PORE_V_P"]
        .sum()
        .sort_values("PTR_P")
    )
    if len(measured_curve) < 4:
        return pd.DataFrame(), None

    positive_positions = np.flatnonzero(measured_curve["PORE_V_P"].to_numpy() > 0)
    if len(positive_positions) < 2:
        return pd.DataFrame(), None

    return measured_curve, "measured PORE_V_P"


def _build_pore_throat_curve(sample_group):
    washburn_curve, washburn_source = _build_washburn_pore_throat_curve(sample_group)
    if len(washburn_curve) >= 2:
        return washburn_curve, washburn_source

    return _build_measured_pore_throat_curve(sample_group)


def _plot_pore_throat_radius_distribution(df, sample_col=None, type_col="AutoPoreType", title=None):
    if "PC_STRESS_CORR" not in df.columns or "SW_STRESS_CORR" not in df.columns:
        return None

    valid = df[
        (df["PC_STRESS_CORR"] > 0)
        & (df["SW_STRESS_CORR"] >= 0)
        & (df["SW_STRESS_CORR"] <= 1)
        & df[type_col].notna()
    ].copy()
    if valid.empty:
        return None

    fig = go.Figure()

    type_values = sorted(
        valid[type_col].dropna().astype(str).unique(),
        key=_type_sort_key
    )

    for type_index, t in enumerate(type_values):
        group = valid[valid[type_col].astype(str) == t].copy()
        base_type = str(t).split(".")[0]
        try:
            base_type_number = int(base_type)
            color = COLOR_MAP.get(
                base_type_number,
                COLOR_MAP.get(((base_type_number - 1) % len(COLOR_MAP)) + 1, "gray")
            )
        except ValueError:
            color = COLOR_MAP.get((type_index % len(COLOR_MAP)) + 1, "gray")

        if sample_col in group.columns:
            sample_groups = group.groupby(sample_col, dropna=False)
        else:
            sample_groups = [(None, group)]

        legend_shown = False
        for _, (sample_id, sample_group) in enumerate(sample_groups):
            sample_curve, curve_source = _build_pore_throat_curve(sample_group)
            if len(sample_curve) < 2:
                continue

            sample_text = str(sample_id) if sample_id is not None else "N/A"
            fig.add_trace(
                go.Scatter(
                    x=sample_curve["PTR_P"],
                    y=sample_curve["PORE_V_P"],
                    mode="lines",
                    line=dict(width=1, color=color),
                    name=f"Type {t}",
                    legendgroup=f"Type {t}",
                    showlegend=not legend_shown,
                    customdata=np.column_stack([
                        np.repeat(sample_text, len(sample_curve)),
                        np.repeat(str(t), len(sample_curve)),
                        np.repeat(curve_source or "N/A", len(sample_curve)),
                    ]),
                    hovertemplate=(
                        "Sample: %{customdata[0]}<br>"
                        "Type: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}<br>"
                        "PTR_P: %{x:.4g}<br>"
                        "PORE_V_P: %{y:.4g}<extra></extra>"
                    )
                )
            )
            legend_shown = True

    fig.update_layout(
        title=title or "Pore Throat Radius Distribution Curves by Autonomous Pore Type",
        xaxis_title="Pore throat radius, PTR_P (um)",
        yaxis_title="PORE_V_P (%)",
        xaxis=dict(type="log")
    )
    _style_figure(fig, legend_title=type_col)

    return fig


def _curve_correspondence_summary(df, sample_col, type_col):
    if sample_col not in df.columns or type_col not in df.columns:
        return pd.DataFrame()

    rows = []
    type_values = sorted(
        df[type_col].dropna().astype(str).unique(),
        key=_type_sort_key
    )
    for t in type_values:
        group = df[df[type_col].astype(str) == t]
        capillary_count = 0
        pore_count = 0
        washburn_count = 0
        fallback_measured_count = 0

        for _, sample_group in group.groupby(sample_col, dropna=False):
            capillary_curve = (
                sample_group[["SW_STRESS_CORR", "PC_STRESS_CORR"]]
                .dropna()
                .groupby("SW_STRESS_CORR", as_index=False)["PC_STRESS_CORR"]
                .median()
            )
            if len(capillary_curve) >= 2:
                capillary_count += 1

            pore_curve, curve_source = _build_pore_throat_curve(sample_group)
            if len(pore_curve) >= 2:
                pore_count += 1
                if curve_source == "Washburn from MICP":
                    washburn_count += 1
                elif str(curve_source).startswith("Washburn from MICP"):
                    washburn_count += 1
                elif curve_source == "measured PORE_V_P":
                    fallback_measured_count += 1

        rows.append({
            "Type": t,
            "Capillary curves": capillary_count,
            "Pore throat curves": pore_count,
            "Washburn MICP curves": washburn_count,
            "Fallback measured PORE_V_P curves": fallback_measured_count,
            "Missing pore throat curves": max(capillary_count - pore_count, 0),
        })

    return pd.DataFrame(rows)


def _plot_poroperm(df):
    fig = go.Figure()

    for t in sorted(df["AutoPoreType"].dropna().unique()):
        group = df[df["AutoPoreType"] == t].copy()
        color = COLOR_MAP.get(int(t), "gray")

        fig.add_trace(
            go.Scatter(
                x=group["CPOR_clean"],
                y=group["CKH_clean"],
                mode="markers",
                marker=dict(size=SCATTER_MARKER_SIZE, color=color, opacity=0.45),
                name=f"Type {int(t)}"
            )
        )

        if len(group) >= 5 and group["CPOR_clean"].min() != group["CPOR_clean"].max():
            x = group["CPOR_clean"].values
            y = np.log10(group["CKH_clean"].values)
            b, a = np.polyfit(x, y, 1)
            phi_fit = np.linspace(group["CPOR_clean"].min(), group["CPOR_clean"].max(), 100)
            k_fit = 10 ** (b * phi_fit + a)

            fig.add_trace(
                go.Scatter(
                    x=phi_fit,
                    y=k_fit,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"Type {int(t)} fit"
                )
            )

    fig.update_layout(
        title="Porosity-Permeability Crossplot with Autonomous Type Fits",
        xaxis_title="CPOR_clean (v/v)",
        yaxis_title="CKH_clean (mD)",
        yaxis=dict(type="log")
    )
    _style_figure(fig, legend_title="Auto Type")

    return fig


def _plot_class_controlled_fzi(
    df,
    fzi_by_type=None,
    type_col="AutoPoreType",
    title="Classification-Controlled FZI Curves",
    porosity_cutoff=None
):
    fig = go.Figure()

    if type_col not in df.columns:
        type_col = "AutoPoreType"

    phi = np.linspace(0.01, 0.4, 200)
    if fzi_by_type is None:
        class_fzi = (
            df.groupby(type_col)["FZI"]
            .median()
            .dropna()
            .sort_index()
            .to_dict()
        )
    else:
        class_fzi = {
            str(type_id): float(fzi_value)
            for type_id, fzi_value in fzi_by_type.items()
            if pd.notna(fzi_value) and float(fzi_value) > 0
        }

    type_values = sorted(
        df[type_col].dropna().astype(str).unique(),
        key=_type_sort_key
    )

    for type_index, t in enumerate(type_values):
        group = df[df[type_col].astype(str) == t].copy()
        color = _color_for_type(t, type_index)
        marker_opacity = 0.30 if str(t) == "Invalid" else 0.42

        fig.add_trace(
            go.Scatter(
                x=group["CPOR_clean"],
                y=group["CKH_clean"],
                mode="markers",
                marker=dict(size=SCATTER_MARKER_SIZE, color=color, opacity=marker_opacity),
                name="Invalid" if str(t) == "Invalid" else f"Type {t}"
            )
        )

    for type_index, (t, fzi_value) in enumerate(sorted(class_fzi.items(), key=lambda item: _type_sort_key(item[0]))):
        k = 1014 * (fzi_value ** 2) * (phi ** 3) / ((1 - phi) ** 2)
        color = _color_for_type(t, type_index)

        fig.add_trace(
            go.Scatter(
                x=phi,
                y=k,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"Type {t} FZI={fzi_value:.2f}"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="CPOR_clean (v/v)",
        yaxis_title="CKH_clean (mD)",
        xaxis=dict(range=[0, 0.4]),
        yaxis=dict(type="log", range=[-3, 4])
    )
    if porosity_cutoff is not None:
        fig.add_vline(
            x=porosity_cutoff,
            line_width=2,
            line_dash="dot",
            line_color=COLORS.get("accent_teal", COLORS["primary"]),
            annotation_text=f"Porosity cutoff {porosity_cutoff:.3f}",
            annotation_position="top left"
        )
    _style_figure(fig, legend_title=f"{type_col} / FZI")

    return fig


def _color_for_type(type_label, type_index=0):
    if str(type_label) == "Invalid":
        return "#7f7f7f"

    base_type = str(type_label).split(".")[0]
    try:
        base_type_number = int(base_type)
        return COLOR_MAP.get(
            base_type_number,
            COLOR_MAP.get(((base_type_number - 1) % len(COLOR_MAP)) + 1, "gray")
        )
    except ValueError:
        return COLOR_MAP.get((type_index % len(COLOR_MAP)) + 1, "gray")


def _plot_rca_fzi_constrained(df, fzi_by_type=None, type_col="AutoPoreType"):
    if type_col not in df.columns:
        type_col = "AutoPoreType"

    manual_fzi = {}
    if fzi_by_type:
        manual_fzi = {
            str(type_id): float(fzi_value)
            for type_id, fzi_value in fzi_by_type.items()
            if pd.notna(fzi_value) and float(fzi_value) > 0
        }

    valid = df[
        (df["CPOR_clean"] > 0) &
        (df["CPOR_clean"] < 1) &
        (df["CKH_clean"] > 0) &
        df[type_col].notna()
    ].copy()
    if valid.empty:
        return None

    fig = go.Figure()
    phi = np.linspace(0.005, 0.35, 260)
    fzi_boundaries = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    for fzi in fzi_boundaries:
        k = 1014 * (fzi ** 2) * (phi ** 3) / ((1 - phi) ** 2)
        fig.add_trace(
            go.Scatter(
                x=phi,
                y=k,
                mode="lines",
                line=dict(color="rgba(150,150,150,0.55)", width=2),
                name=f"FZI {fzi:g}",
                showlegend=False,
                hovertemplate=f"FZI={fzi:g}<br>CPOR=%{{x:.3f}}<br>CKH=%{{y:.3g}}<extra></extra>",
            )
        )

    type_values = sorted(
        valid[type_col].dropna().astype(str).unique(),
        key=_type_sort_key
    )

    for type_index, t in enumerate(type_values):
        group = valid[valid[type_col].astype(str) == t].copy()
        color = _color_for_type(t, type_index)

        fig.add_trace(
            go.Scatter(
                x=group["CPOR_clean"],
                y=group["CKH_clean"],
                mode="markers",
                marker=dict(size=SCATTER_MARKER_SIZE, color=color, opacity=0.58),
                name=f"Type {t}",
                legendgroup=f"Type {t}",
            )
        )

        if len(group) < 5 or group["CPOR_clean"].nunique() < 2:
            continue

        x = group["CPOR_clean"].to_numpy()
        y = np.log10(group["CKH_clean"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)

        phi_fit = np.linspace(x.min(), x.max(), 120)
        k_fit = 10 ** (slope * phi_fit + intercept)

        fzi_center = manual_fzi.get(str(t), group["FZI"].median())
        lower_fzi = fzi_center * 0.75
        upper_fzi = fzi_center * 1.25
        lower_curve = 1014 * (lower_fzi ** 2) * (phi_fit ** 3) / ((1 - phi_fit) ** 2)
        upper_curve = 1014 * (upper_fzi ** 2) * (phi_fit ** 3) / ((1 - phi_fit) ** 2)
        k_fit = np.clip(k_fit, lower_curve, upper_curve)

        fig.add_trace(
            go.Scatter(
                x=phi_fit,
                y=k_fit,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"Type {t} RCA fit",
                legendgroup=f"Type {t}",
                hovertemplate=(
                    f"Type {t} RCA fit<br>"
                    f"FZI constraint={fzi_center:.2f}<br>"
                    "CPOR=%{x:.3f}<br>CKH=%{y:.3g}<extra></extra>"
                ),
            )
        )

    y_min = max(valid["CKH_clean"].min() * 0.3, 1e-4)
    y_max = valid["CKH_clean"].max() * 3
    fig.update_layout(
        title="RCA FZI-Constrained Classification Crossplot",
        xaxis_title="CPOR_clean (v/v)",
        yaxis_title="CKH_clean (mD)",
        xaxis=dict(range=[0, max(0.35, valid["CPOR_clean"].max() * 1.05)]),
        yaxis=dict(type="log", range=[np.log10(y_min), np.log10(y_max)])
    )
    _style_figure(fig, legend_title=f"{type_col} / RCA")
    fig.update_xaxes(mirror=True, showline=True, linecolor=COLORS["ink"])
    fig.update_yaxes(mirror=True, showline=True, linecolor=COLORS["ink"])

    return fig


def _plot_feature_space(sample_features):
    if "PCA1" not in sample_features.columns or "PCA2" not in sample_features.columns:
        return None

    fig = go.Figure()

    for t in sorted(sample_features["AutoPoreType"].dropna().unique()):
        group = sample_features[sample_features["AutoPoreType"] == t]
        fig.add_trace(
            go.Scatter(
                x=group["PCA1"],
                y=group["PCA2"],
                mode="markers",
                marker=dict(size=FEATURE_MARKER_SIZE, color=COLOR_MAP.get(int(t), "gray"), opacity=0.65),
                name=f"Type {int(t)}"
            )
        )

    fig.update_layout(
        title="Autonomous Classification Feature Space (PCA)",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2"
    )
    _style_figure(fig, legend_title="Auto Type")

    return fig


def _summary_table(df):
    summary = (
        df.groupby("AutoPoreType")
        .agg(
            Count=("AutoPoreType", "size"),
            CPOR_median=("CPOR_clean", "median"),
            CKH_median=("CKH_clean", "median"),
            FZI_median=("FZI", "median"),
            R35_median=("R35_equiv", "median"),
            Pc_median=("PC_STRESS_CORR", "median"),
            Sw_median=("SW_STRESS_CORR", "median"),
        )
        .reset_index()
        .rename(columns={"AutoPoreType": "Type"})
    )
    return summary


def _safe_widget_key(value):
    return "".join(
        char if char.isalnum() else "_"
        for char in str(value)
    )


def _render_manual_fzi_inputs(df, type_col="AutoPoreType"):
    if type_col not in df.columns:
        type_col = "AutoPoreType"

    default_fzi = (
        df.groupby(type_col)["FZI"]
        .median()
        .dropna()
    )
    default_fzi = default_fzi.loc[
        sorted(default_fzi.index, key=_type_sort_key)
    ]
    if default_fzi.empty:
        return {}

    st.markdown("### Manual FZI settings")
    st.caption(
        "Enter the FZI value used to draw each classification-controlled curve. "
        "Defaults are filled from the current type median FZI, but the plotted curves use your inputs."
    )

    fzi_by_type = {}
    columns = st.columns(min(4, len(default_fzi)))
    for index, (type_id, default_value) in enumerate(default_fzi.items()):
        with columns[index % len(columns)]:
            type_label = str(type_id)
            fzi_by_type[type_label] = st.number_input(
                f"Type {type_label} FZI",
                min_value=0.0001,
                max_value=1000.0,
                value=float(default_value),
                step=max(float(default_value) * 0.05, 0.01),
                format="%.4f",
                key=f"manual_fzi_{type_col}_{_safe_widget_key(type_label)}"
            )

    return fzi_by_type


def _apply_fzi_validity_screen(df, type_col, porosity_cutoff, fzi_bounds_by_type):
    screened = df.copy()
    screened["ValidityType"] = screened[type_col].astype(str)

    invalid_mask = screened["CPOR_clean"] < porosity_cutoff
    for type_label, bounds in fzi_bounds_by_type.items():
        lower_fzi, upper_fzi = bounds
        type_mask = screened[type_col].astype(str) == str(type_label)
        invalid_mask = invalid_mask | (
            type_mask
            & (
                (screened["FZI"] < lower_fzi)
                | (screened["FZI"] > upper_fzi)
            )
        )

    screened.loc[invalid_mask, "ValidityType"] = "Invalid"
    return screened


def _render_fzi_validity_controls(df, type_col, fzi_by_type):
    valid = df[
        (df["CPOR_clean"] > 0)
        & (df["CPOR_clean"] < 1)
        & (df["FZI"] > 0)
        & df[type_col].notna()
    ].copy()
    if valid.empty:
        st.warning("Valid CPOR_clean and FZI values are required for FZI screening.")
        return df.copy(), {}, 0.0

    st.markdown("### FZI validity screening")
    porosity_min = float(valid["CPOR_clean"].min())
    porosity_max = float(valid["CPOR_clean"].max())
    default_cutoff = max(0.0, min(0.05, porosity_max))
    porosity_cutoff = st.number_input(
        "Porosity cutoff",
        min_value=0.0,
        max_value=max(0.99, porosity_max),
        value=default_cutoff,
        step=0.005,
        format="%.4f",
        key=f"fzi_validity_porosity_cutoff_{type_col}"
    )
    st.caption(
        "Points with CPOR_clean below this cutoff are assigned to Invalid. "
        "For points to the right of the cutoff, each type is screened by its FZI lower and upper bounds."
    )

    type_values = sorted(
        valid[type_col].dropna().astype(str).unique(),
        key=_type_sort_key
    )
    fzi_bounds_by_type = {}
    bound_cols = st.columns(2)
    for index, type_label in enumerate(type_values):
        group = valid[valid[type_col].astype(str) == type_label]
        default_center = fzi_by_type.get(
            type_label,
            float(group["FZI"].median())
        )
        default_lower = max(0.0001, default_center * 0.75)
        default_upper = max(default_lower + 0.0001, default_center * 1.25)
        step_size = max(default_center * 0.05, 0.01)

        with bound_cols[index % len(bound_cols)]:
            st.markdown(f"**Type {type_label} FZI bounds**")
            lower_fzi = st.number_input(
                f"Type {type_label} lower FZI",
                min_value=0.0001,
                max_value=1000.0,
                value=float(default_lower),
                step=step_size,
                format="%.4f",
                key=f"fzi_validity_lower_{type_col}_{_safe_widget_key(type_label)}"
            )
            upper_fzi = st.number_input(
                f"Type {type_label} upper FZI",
                min_value=0.0001,
                max_value=1000.0,
                value=float(default_upper),
                step=step_size,
                format="%.4f",
                key=f"fzi_validity_upper_{type_col}_{_safe_widget_key(type_label)}"
            )
            if upper_fzi <= lower_fzi:
                st.warning(f"Type {type_label}: upper FZI must be greater than lower FZI.")
                upper_fzi = lower_fzi + 0.0001
            fzi_bounds_by_type[type_label] = (lower_fzi, upper_fzi)

    screened = _apply_fzi_validity_screen(
        valid,
        type_col,
        porosity_cutoff,
        fzi_bounds_by_type
    )
    invalid_count = int((screened["ValidityType"] == "Invalid").sum())
    st.caption(
        f"{invalid_count} of {len(screened)} point(s) are assigned to Invalid "
        "after porosity and FZI-bound screening."
    )
    return screened, fzi_bounds_by_type, porosity_cutoff


def run():
    st.header("Autonomous Carbonate Pore Throat Classification")

    st.info(
        "Carbonate-focused version: this module classifies unlabelled pore throat data "
        "using porosity-permeability features plus mercury-injection capillary-pressure "
        "curve morphology. The added carbonate morphology indicators capture entry, "
        "middle, tail, span, and curve-complexity behavior."
    )

    st.markdown("""
    Required columns:
    - `CPOR_clean`: porosity in fraction, such as 0.18
    - `CKH_clean`: permeability in mD
    - `PC_STRESS_CORR`: capillary pressure, interpreted as bar for Washburn pore-throat conversion
    - `SW_STRESS_CORR`: saturation fraction between 0 and 1

    Optional columns:
    - `PTR_P`: pore throat radius, used only as a fallback/reference when MICP conversion is unavailable
    - `PORE_V_P`: pore volume, used only as a fallback/reference when MICP conversion is unavailable
    - sample identifier, such as `SampleID`, `Plug`, `wellName`, or `Well`
    """)

    uploaded_file = st.file_uploader(
        "Upload unclassified pore throat data",
        type=["xlsx", "csv"],
        key="autonomous_pore_upload"
    )

    if uploaded_file is None:
        st.warning("Please upload an unclassified dataset.")
        return

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file, skiprows=[1])
    except Exception:
        uploaded_file.seek(0)
        df_raw = pd.read_excel(uploaded_file)

    try:
        df = _clean_input(df_raw)
        df = _add_petrophysical_features(df)
    except Exception as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.error("No valid data after cleaning. Please check numeric ranges and required columns.")
        return

    df, formation_col, selected_formations = _render_formation_filter(df)
    if df.empty:
        st.error("No valid data remains after formation filtering.")
        return

    try:
        df, sample_col, sample_message = _auto_assign_sample_identifier(df)
    except Exception as exc:
        st.error(str(exc))
        return

    if sample_message:
        st.info(sample_message)

    try:
        sample_features, sample_col, feature_cols = _build_sample_feature_table(df, sample_col)
    except Exception as exc:
        st.error(str(exc))
        return

    with st.sidebar:
        st.markdown("### Autonomous Classification Settings")
        classification_mode = st.radio(
            "Classification mode",
            options=["Automatic", "Manual"],
            horizontal=True,
            key="auto_classification_mode"
        )
        requested_clusters = None
        if classification_mode == "Automatic":
            st.caption(
                "The number of pore types is selected automatically from the data "
                "using capillary-curve feature separation and cluster balance."
            )
        else:
            max_manual_clusters = min(AUTO_CLUSTER_MAX, len(sample_features))
            requested_clusters = st.slider(
                "Number of autonomous pore types",
                min_value=AUTO_CLUSTER_MIN,
                max_value=max_manual_clusters,
                value=min(5, max_manual_clusters),
                step=1,
                key="auto_type_count"
            )
        random_state = st.number_input(
            "Random state",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="auto_random_state"
        )

    try:
        sample_features, _, _, cluster_diagnostics = _cluster_samples(
            sample_features,
            feature_cols,
            int(random_state),
            n_clusters=requested_clusters
        )
        df_classified = _assign_types_to_rows(df, sample_features, sample_col)
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(
        f"Autonomous classification completed: "
        f"{df_classified['AutoPoreType'].nunique()} pore types generated."
    )
    if classification_mode == "Automatic":
        max_candidate_count = (
            AUTO_CLUSTER_MIN
            if len(sample_features) == AUTO_CLUSTER_MIN
            else min(AUTO_CLUSTER_MAX, len(sample_features) - 1)
        )
        st.info(
            f"Automatic cluster selection chose {df_classified['AutoPoreType'].nunique()} "
            f"pore types from {AUTO_CLUSTER_MIN}-{max_candidate_count} "
            "candidate groups."
        )
    else:
        st.info(
            f"Manual cluster selection used {df_classified['AutoPoreType'].nunique()} "
            "pore types."
        )

    st.markdown("### Classified Data Preview")
    preview_cols = [
        col for col in [
            sample_col, formation_col, "CPOR_clean", "CKH_clean", "PC_STRESS_CORR",
            "SW_STRESS_CORR", "PTR_P", "FZI", "R35_equiv", "AutoPoreType"
        ] if col and col in df_classified.columns
    ]
    st.dataframe(df_classified[preview_cols].head(200), use_container_width=True)

    st.markdown("### Classification Summary")
    summary = _summary_table(df_classified)
    st.dataframe(summary, use_container_width=True)

    csv = df_classified.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download classified data as CSV",
        data=csv,
        file_name="autonomous_pore_typing_result.csv",
        mime="text/csv"
    )

    tab1, tab2, tab3 = st.tabs([
        "Capillary Curves & Pore Throat Radius",
        "Classification-Controlled FZI",
        "RCA FZI-Constrained Classification"
    ])

    with tab1:
        st.markdown("### Uncorrected classification")
        raw_cols = st.columns(2)
        with raw_cols[0]:
            st.plotly_chart(
                _plot_capillary_curves(df_classified, sample_col),
                use_container_width=True
            )
        with raw_cols[1]:
            raw_radius_fig = _plot_pore_throat_radius_distribution(
                df_classified,
                sample_col,
                title="Pore Throat Radius Distribution Curves by Autonomous Pore Type"
            )
            if raw_radius_fig is not None:
                st.plotly_chart(raw_radius_fig, use_container_width=True)
            else:
                st.warning("Valid PC_STRESS_CORR and SW_STRESS_CORR values are required to draw Washburn pore throat curves.")

        st.caption(
            "Carbonate version: each sample is drawn as a continuous log-Pc interpolated curve. "
            "The classification includes capillary-pressure shape metrics such as entry/tail "
            "pressure, curve span, segment slopes, and slope complexity."
        )
        correspondence_summary = _curve_correspondence_summary(
            df_classified,
            sample_col,
            "AutoPoreType"
        )
        if not correspondence_summary.empty:
            with st.expander("Curve correspondence check", expanded=True):
                st.dataframe(correspondence_summary, use_container_width=True)
                st.caption(
                    "If pore throat curves are fewer than capillary curves, the missing samples "
                    "do not have enough pressure-saturation points for Washburn conversion. "
                    "The pore throat plot is derived from MICP first; measured PTR_P/PORE_V_P "
                    "is used only as a fallback."
                )

        st.markdown("### Manual classification correction")
        df_corrected, selected_sample_key = _render_manual_type_correction_controls(
            df_classified,
            sample_features,
            sample_col,
            key_prefix="manual_combined"
        )

        selected_curve = df_corrected[
            df_corrected[sample_col].astype(str) == selected_sample_key
        ].copy()
        st.markdown("### Selected curve")
        selected_cols = st.columns(2)
        with selected_cols[0]:
            st.plotly_chart(
                _plot_capillary_curves(
                    selected_curve,
                    sample_col,
                    type_col="CorrectedPoreType",
                    title="Selected Capillary Pressure Curve"
                ),
                use_container_width=True
            )
        with selected_cols[1]:
            selected_radius_fig = _plot_pore_throat_radius_distribution(
                selected_curve,
                sample_col,
                type_col="CorrectedPoreType",
                title="Selected Pore Throat Radius Distribution Curve"
            )
            if selected_radius_fig is not None:
                st.plotly_chart(selected_radius_fig, use_container_width=True)
            else:
                st.warning("The selected curve does not have enough pressure-saturation points for Washburn conversion.")

        curve_type_col = st.radio(
            "Correction result grouping",
            options=["CorrectedPoreType", "AutoPoreType"],
            horizontal=True,
            key="manual_combined_type_col"
        )
        st.markdown("### Correction result")
        corrected_cols = st.columns(2)
        with corrected_cols[0]:
            st.plotly_chart(
                _plot_capillary_curves(
                    df_corrected,
                    sample_col,
                    type_col=curve_type_col,
                    title=f"Capillary Pressure Curves by {curve_type_col}"
                ),
                use_container_width=True
            )
        with corrected_cols[1]:
            radius_fig = _plot_pore_throat_radius_distribution(
                df_corrected,
                sample_col,
                type_col=curve_type_col,
                title=f"Pore Throat Radius Distribution Curves by {curve_type_col}"
            )
            if radius_fig is not None:
                st.plotly_chart(radius_fig, use_container_width=True)
            else:
                st.warning("Valid PC_STRESS_CORR and SW_STRESS_CORR values are required to draw Washburn pore throat curves.")

    with tab2:
        fzi_source = df_corrected if "df_corrected" in locals() else df_classified
        fzi_type_col = "CorrectedPoreType" if "CorrectedPoreType" in fzi_source.columns else "AutoPoreType"
        manual_fzi_by_type = _render_manual_fzi_inputs(fzi_source, type_col=fzi_type_col)
        st.plotly_chart(
            _plot_class_controlled_fzi(
                fzi_source,
                manual_fzi_by_type,
                type_col=fzi_type_col
            ),
            use_container_width=True
        )
        st.caption(
            "The FZI curves in this panel are controlled by manually entered FZI values "
            "for each corrected pore type."
        )
        screened_fzi_source, _, porosity_cutoff = _render_fzi_validity_controls(
            fzi_source,
            fzi_type_col,
            manual_fzi_by_type
        )
        screened_valid_types = {
            type_label
            for type_label in screened_fzi_source["ValidityType"].dropna().astype(str).unique()
            if type_label != "Invalid"
        }
        screened_fzi_by_type = {
            type_label: fzi_value
            for type_label, fzi_value in manual_fzi_by_type.items()
            if str(type_label) in screened_valid_types
        }
        st.plotly_chart(
            _plot_class_controlled_fzi(
                screened_fzi_source,
                screened_fzi_by_type,
                type_col="ValidityType",
                title="FZI Validity-Screened Classification",
                porosity_cutoff=porosity_cutoff
            ),
            use_container_width=True
        )

    with tab3:
        rca_fig = _plot_rca_fzi_constrained(
            fzi_source,
            manual_fzi_by_type,
            type_col=fzi_type_col
        )
        if rca_fig is not None:
            st.plotly_chart(rca_fig, use_container_width=True)
        else:
            st.warning("Valid CPOR_clean and CKH_clean values are required to draw the RCA crossplot.")
