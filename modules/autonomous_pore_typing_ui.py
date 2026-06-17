import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


REQUIRED_COLUMNS = ["CPOR_clean", "CKH_clean", "PC_STRESS_CORR", "SW_STRESS_CORR"]
OPTIONAL_COLUMNS = ["PTR_P", "PORE_V_P"]
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

COLOR_MAP = {
    1: "blue",
    2: "green",
    3: "goldenrod",
    4: "red",
    5: "orange",
    6: "purple",
    7: "brown",
    8: "black",
}

SCATTER_MARKER_SIZE = 4
FEATURE_MARKER_SIZE = 6
CURVE_INTERPOLATION_POINTS = 160
CARBONATE_PC_SATURATIONS = [0.10, 0.25, 0.50, 0.75, 0.90]
PORE_RADIUS_BINS = 40


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


def _cluster_samples(sample_features, feature_cols, n_clusters, random_state):
    n_clusters = int(min(n_clusters, len(sample_features)))
    if n_clusters < 2:
        raise ValueError(
            f"At least two valid samples are required for autonomous classification; "
            f"only {len(sample_features)} valid sample is available."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_features[feature_cols])

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

    return sample_features, model, scaler


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
            color = COLOR_MAP.get(int(base_type), "gray")
        except ValueError:
            color = COLOR_MAP.get((type_index % len(COLOR_MAP)) + 1, "gray")

        if sample_col in group.columns:
            sample_groups = group.groupby(sample_col, dropna=False)
        else:
            sample_groups = [(None, group)]

        for i, (_, sample_group) in enumerate(sample_groups):
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

            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode="lines",
                    line=dict(width=1, color=color),
                    name=f"Type {t}",
                    legendgroup=f"Type {t}",
                    showlegend=(i == 0)
                )
            )

    fig.update_layout(
        title=title or "Mercury Injection Capillary Pressure Curves by Autonomous Pore Type",
        xaxis_title="Water Saturation / Mercury Saturation Proxy (fraction)",
        yaxis_title="Capillary Pressure (Pc)",
        yaxis=dict(type="log"),
        plot_bgcolor="white",
        legend=dict(title=type_col)
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


def _plot_pore_throat_radius_distribution(df, sample_col=None, type_col="AutoPoreType"):
    if "PTR_P" not in df.columns or "PORE_V_P" not in df.columns:
        return None

    valid = df[
        (df["PTR_P"] > 0) &
        (df["PORE_V_P"] > 0) &
        df[type_col].notna()
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
            color = COLOR_MAP.get(int(base_type), "gray")
        except ValueError:
            color = COLOR_MAP.get((type_index % len(COLOR_MAP)) + 1, "gray")

        if sample_col in group.columns:
            sample_groups = group.groupby(sample_col, dropna=False)
        else:
            sample_groups = [(None, group)]

        for i, (_, sample_group) in enumerate(sample_groups):
            sample_curve = (
                sample_group[["PTR_P", "PORE_V_P"]]
                .dropna()
                .groupby("PTR_P", as_index=False)["PORE_V_P"]
                .sum()
                .sort_values("PTR_P")
            )
            if len(sample_curve) < 2:
                continue

            fig.add_trace(
                go.Scatter(
                    x=sample_curve["PTR_P"],
                    y=sample_curve["PORE_V_P"],
                    mode="lines",
                    line=dict(width=1, color=color),
                    name=f"Type {t}",
                    legendgroup=f"Type {t}",
                    showlegend=(i == 0)
                )
            )

    fig.update_layout(
        title="Pore Throat Radius Distribution Curves by Autonomous Pore Type",
        xaxis_title="Pore throat radius, PTR_P (um)",
        yaxis_title="PORE_V_P (%)",
        xaxis=dict(type="log"),
        plot_bgcolor="white",
        legend=dict(title=type_col)
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


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
        yaxis=dict(type="log"),
        plot_bgcolor="white",
        legend=dict(title="Auto Type")
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


def _plot_class_controlled_fzi(df):
    fig = go.Figure()

    phi = np.linspace(0.01, 0.4, 200)
    class_fzi = (
        df.groupby("AutoPoreType")["FZI"]
        .median()
        .dropna()
        .sort_index()
    )

    for t in sorted(df["AutoPoreType"].dropna().unique()):
        group = df[df["AutoPoreType"] == t].copy()
        color = COLOR_MAP.get(int(t), "gray")

        fig.add_trace(
            go.Scatter(
                x=group["CPOR_clean"],
                y=group["CKH_clean"],
                mode="markers",
                marker=dict(size=SCATTER_MARKER_SIZE, color=color, opacity=0.42),
                name=f"Type {int(t)}"
            )
        )

    for t, fzi_value in class_fzi.items():
        k = 1014 * (fzi_value ** 2) * (phi ** 3) / ((1 - phi) ** 2)
        color = COLOR_MAP.get(int(t), "gray")

        fig.add_trace(
            go.Scatter(
                x=phi,
                y=k,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"Type {int(t)} median FZI={fzi_value:.2f}"
            )
        )

    fig.update_layout(
        title="Classification-Controlled FZI Curves",
        xaxis_title="CPOR_clean (v/v)",
        yaxis_title="CKH_clean (mD)",
        xaxis=dict(range=[0, 0.4]),
        yaxis=dict(type="log", range=[-3, 4]),
        plot_bgcolor="white",
        legend=dict(title="Auto Type / FZI")
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

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
        yaxis_title="PCA 2",
        plot_bgcolor="white",
        legend=dict(title="Auto Type")
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

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
    - `PC_STRESS_CORR`: capillary pressure
    - `SW_STRESS_CORR`: saturation fraction between 0 and 1

    Optional columns:
    - `PTR_P`: pore throat radius
    - `PORE_V_P`: pore volume, used as the weight for pore throat radius distribution when available
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

    try:
        df, sample_col, sample_message = _auto_assign_sample_identifier(df)
    except Exception as exc:
        st.error(str(exc))
        return

    if sample_message:
        st.info(sample_message)

    with st.sidebar:
        st.markdown("### Autonomous Classification Settings")
        requested_clusters = st.slider(
            "Number of autonomous pore types",
            min_value=2,
            max_value=8,
            value=min(5, max(2, len(df))),
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
        sample_features, sample_col, feature_cols = _build_sample_feature_table(df, sample_col)
        sample_features, _, _ = _cluster_samples(
            sample_features,
            feature_cols,
            requested_clusters,
            int(random_state)
        )
        df_classified = _assign_types_to_rows(df, sample_features, sample_col)
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(
        f"Autonomous classification completed: "
        f"{df_classified['AutoPoreType'].nunique()} pore types generated."
    )

    st.markdown("### Classified Data Preview")
    preview_cols = [
        col for col in [
            sample_col, "CPOR_clean", "CKH_clean", "PC_STRESS_CORR",
            "SW_STRESS_CORR", "PTR_P", "FZI", "R35_equiv", "AutoPoreType"
        ] if col in df_classified.columns
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Carbonate Capillary Pressure Curves",
        "Pore Throat Radius Distribution",
        "Porosity-Permeability Plot",
        "Classification-Controlled FZI",
        "Feature Space"
    ])

    with tab1:
        st.plotly_chart(_plot_capillary_curves(df_classified, sample_col), use_container_width=True)
        st.caption(
            "Carbonate version: each sample is drawn as a continuous log-Pc interpolated curve. "
            "The classification includes capillary-pressure shape metrics such as entry/tail "
            "pressure, curve span, segment slopes, and slope complexity."
        )

        st.markdown("### Secondary classification and merge")

        available_types = sorted(df_classified["AutoPoreType"].dropna().unique())
        selected_parent_type = st.selectbox(
            "Primary type to split",
            options=available_types,
            format_func=lambda value: f"Type {int(value)}",
            key="secondary_parent_type"
        )

        parent_sample_count = int(
            (sample_features["AutoPoreType"] == selected_parent_type).sum()
        )
        max_subtypes = min(6, parent_sample_count)

        if parent_sample_count < 2:
            st.warning(
                f"Type {int(selected_parent_type)} has only {parent_sample_count} sample, "
                "so it cannot be split further."
            )
            refined_features = sample_features.copy()
            refined_features["RefinedPoreType"] = refined_features["AutoPoreType"].astype(int).astype(str)
            refined_features["FinalPoreType"] = refined_features["RefinedPoreType"]
        else:
            requested_subtypes = st.slider(
                "Number of secondary subtypes",
                min_value=2,
                max_value=max_subtypes,
                value=min(2, max_subtypes),
                step=1,
                key="secondary_type_count"
            )

            try:
                refined_features = _subcluster_selected_type(
                    sample_features,
                    feature_cols,
                    selected_parent_type,
                    requested_subtypes,
                    int(random_state)
                )
            except Exception as exc:
                st.error(str(exc))
                refined_features = sample_features.copy()
                refined_features["RefinedPoreType"] = refined_features["AutoPoreType"].astype(int).astype(str)
                refined_features["FinalPoreType"] = refined_features["RefinedPoreType"]

        refined_labels = sorted(
            refined_features["RefinedPoreType"].dropna().astype(str).unique(),
            key=_type_sort_key
        )

        merge_rule_count = st.number_input(
            "Number of merge rules",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            key="merge_rule_count"
        )

        merge_rules = []
        for i in range(int(merge_rule_count)):
            cols = st.columns([2, 1])
            with cols[0]:
                selected_labels = st.multiselect(
                    f"Types to merge #{i + 1}",
                    options=refined_labels,
                    key=f"merge_labels_{i}"
                )
            with cols[1]:
                default_label = f"M{i + 1}"
                new_label = st.text_input(
                    f"New type #{i + 1}",
                    value=default_label,
                    key=f"merge_name_{i}"
                )
            merge_rules.append((selected_labels, new_label))

        refined_features = _apply_merge_rules(refined_features, merge_rules)
        df_refined = _assign_refined_types_to_rows(df_classified, refined_features, sample_col)

        curve_type_col = st.radio(
            "Curve grouping",
            options=["FinalPoreType", "RefinedPoreType", "AutoPoreType"],
            horizontal=True,
            key="secondary_curve_type_col"
        )
        st.plotly_chart(
            _plot_capillary_curves(
                df_refined,
                sample_col,
                type_col=curve_type_col,
                title=f"Capillary Pressure Curves by {curve_type_col}"
            ),
            use_container_width=True
        )

    with tab2:
        radius_fig = _plot_pore_throat_radius_distribution(df_classified, sample_col)
        if radius_fig is not None:
            st.plotly_chart(radius_fig, use_container_width=True)
        else:
            st.warning("PTR_P and positive PORE_V_P values are required to draw pore throat distribution curves.")

    with tab3:
        st.plotly_chart(_plot_poroperm(df_classified), use_container_width=True)

    with tab4:
        st.plotly_chart(_plot_class_controlled_fzi(df_classified), use_container_width=True)
        st.caption(
            "The FZI curves in this panel are controlled by the autonomous classification: "
            "each curve uses the median FZI of its classified pore type."
        )

    with tab5:
        feature_fig = _plot_feature_space(sample_features)
        if feature_fig is not None:
            st.plotly_chart(feature_fig, use_container_width=True)
        st.markdown("### Features used for autonomous classification")
        st.write(feature_cols)
        st.dataframe(sample_features.head(200), use_container_width=True)
