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
    "wellName", "Well", "WELL", "Core", "CoreNo"
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


def _find_sample_col(df):
    for col in SAMPLE_ID_CANDIDATES:
        if col in df.columns:
            return col
    return None


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
    df["FZI"] = 0.0314 * np.sqrt(k / phi) / (1 - phi)
    df["logFZI"] = np.log10(df["FZI"])
    df["R35_equiv"] = k * (1 - phi) / phi
    df["logR35_equiv"] = np.log10(df["R35_equiv"])

    if "PTR_P" in df.columns:
        df["logPTR"] = np.where(df["PTR_P"] > 0, np.log10(df["PTR_P"]), np.nan)

    return df


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

    feature_cols = [
        "CPOR_clean_median",
        "logK_median",
        "logFZI_median",
        "logR35_equiv_median",
        "logPc_median",
        "logPc_max",
        "SW_STRESS_CORR_min",
        "SW_STRESS_CORR_median",
        "SW_STRESS_CORR_max",
    ]

    if "logPTR_median" in sample_features.columns:
        feature_cols.append("logPTR_median")
    if "PORE_V_P_median" in sample_features.columns:
        feature_cols.append("PORE_V_P_median")

    sample_features = sample_features.dropna(subset=feature_cols).copy()

    return sample_features, sample_col, feature_cols


def _cluster_samples(sample_features, feature_cols, n_clusters, random_state):
    n_clusters = int(min(n_clusters, len(sample_features)))
    if n_clusters < 2:
        raise ValueError("At least two valid samples are required for autonomous classification.")

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


def _plot_capillary_curves(df):
    fig = go.Figure()

    for t in sorted(df["AutoPoreType"].dropna().unique()):
        group = df[df["AutoPoreType"] == t].copy()
        group = group.sort_values("SW_STRESS_CORR")

        fig.add_trace(
            go.Scatter(
                x=group["SW_STRESS_CORR"],
                y=group["PC_STRESS_CORR"],
                mode="lines+markers",
                marker=dict(size=4, color=COLOR_MAP.get(int(t), "gray"), opacity=0.55),
                line=dict(width=2, color=COLOR_MAP.get(int(t), "gray")),
                name=f"Type {int(t)}"
            )
        )

    fig.update_layout(
        title="Mercury Injection Capillary Pressure Curves by Autonomous Pore Type",
        xaxis_title="Water Saturation / Mercury Saturation Proxy (fraction)",
        yaxis_title="Capillary Pressure (Pc)",
        yaxis=dict(type="log"),
        plot_bgcolor="white",
        legend=dict(title="Auto Type")
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
                marker=dict(size=5, color=color, opacity=0.55),
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
                marker=dict(size=5, color=color, opacity=0.5),
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
                marker=dict(size=8, color=COLOR_MAP.get(int(t), "gray"), opacity=0.75),
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
    st.header("Autonomous Pore Throat Classification")

    st.info(
        "This module classifies unlabelled pore throat data using porosity-permeability "
        "and mercury-injection capillary-pressure features. The resulting autonomous "
        "types are then used to generate class-controlled FZI curves."
    )

    st.markdown("""
    Required columns:
    - `CPOR_clean`: porosity in fraction, such as 0.18
    - `CKH_clean`: permeability in mD
    - `PC_STRESS_CORR`: capillary pressure
    - `SW_STRESS_CORR`: saturation fraction between 0 and 1

    Optional columns:
    - `PTR_P`: pore throat radius
    - `PORE_V_P`: pore volume
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

    sample_col = _find_sample_col(df)

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

    tab1, tab2, tab3, tab4 = st.tabs([
        "Capillary Pressure Curves",
        "Porosity-Permeability Plot",
        "Classification-Controlled FZI",
        "Feature Space"
    ])

    with tab1:
        st.plotly_chart(_plot_capillary_curves(df_classified), use_container_width=True)

    with tab2:
        st.plotly_chart(_plot_poroperm(df_classified), use_container_width=True)

    with tab3:
        st.plotly_chart(_plot_class_controlled_fzi(df_classified), use_container_width=True)
        st.caption(
            "The FZI curves in this panel are controlled by the autonomous classification: "
            "each curve uses the median FZI of its classified pore type."
        )

    with tab4:
        feature_fig = _plot_feature_space(sample_features)
        if feature_fig is not None:
            st.plotly_chart(feature_fig, use_container_width=True)
        st.markdown("### Features used for autonomous classification")
        st.write(feature_cols)
        st.dataframe(sample_features.head(200), use_container_width=True)
