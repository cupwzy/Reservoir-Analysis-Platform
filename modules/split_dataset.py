import os

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "CKH_clean",
    "CPOR_clean",
    "PTR_P",
    "PC_STRESS_CORR",
    "SW_STRESS_CORR",
]

OPTIONAL_COLS = [
    "PORE_V_P",
    "wellName",
    "WellName_2",
    "Plug No",
    "ID",
    "ReferenceName",
]


def _first_available_column(df, candidates):
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _build_core_id(df):
    df = df.copy()

    if {"WellName_2", "Plug No"}.issubset(df.columns):
        well = df["WellName_2"].astype(str).str.strip()
        plug = df["Plug No"].astype(str).str.strip()
        df["Core_ID"] = well + "_" + plug
        return df

    core_col = _first_available_column(
        df,
        ["Core_ID", "SampleID", "Sample_ID", "Sample", "Plug", "Plug_ID", "ID", "ReferenceName"],
    )
    if core_col is None:
        raise ValueError(
            "No core/plug identifier column found. Add WellName_2 + Plug No, Core_ID, "
            "SampleID, Plug_ID, ID, or ReferenceName before splitting."
        )

    df["Core_ID"] = df[core_col].astype(str).str.strip()
    return df


def _split_by_core(df):
    core_labels = (
        df.groupby("Core_ID", as_index=False)
        .agg(Label=("Label", lambda s: s.mode().iat[0]))
    )

    train_cores, temp_cores = train_test_split(
        core_labels,
        test_size=0.3,
        random_state=42,
        stratify=core_labels["Label"],
    )

    val_cores, test_cores = train_test_split(
        temp_cores,
        test_size=0.5,
        random_state=42,
        stratify=temp_cores["Label"],
    )

    train_ids = set(train_cores["Core_ID"])
    val_ids = set(val_cores["Core_ID"])
    test_ids = set(test_cores["Core_ID"])

    df_train = df[df["Core_ID"].isin(train_ids)].copy()
    df_val = df[df["Core_ID"].isin(val_ids)].copy()
    df_test = df[df["Core_ID"].isin(test_ids)].copy()

    return df_train, df_val, df_test


def split_data(file_path):
    df = pd.read_excel(file_path, skiprows=[1])
    print(f"Raw data shape: {df.shape}")

    df = df[pd.to_numeric(df["Labelling260205"], errors="coerce").notna()].copy()
    df["Label"] = df["Labelling260205"].astype(int)
    df = df[df["Label"].between(1, 8)].copy()
    print(f"After label clean: {df.shape}")

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _build_core_id(df)
    df = df.dropna(subset=FEATURE_COLS + ["Core_ID"]).copy()

    keep_cols = FEATURE_COLS + ["Label", "Core_ID"]
    keep_cols.extend([col for col in OPTIONAL_COLS if col in df.columns and col not in keep_cols])
    df = df[keep_cols].copy()

    core_sizes = df.groupby("Core_ID").size()
    valid_curve_count = int((core_sizes >= 2).sum())
    print(f"Final dataset: {df.shape}")
    print(f"Core_ID groups: {len(core_sizes)} total, {valid_curve_count} with at least 2 rows")

    df_train, df_val, df_test = _split_by_core(df)

    print("\nDataset Split:")
    print(f"Train: {df_train.shape}, Core_ID: {df_train['Core_ID'].nunique()}")
    print(f"Validation: {df_val.shape}, Core_ID: {df_val['Core_ID'].nunique()}")
    print(f"Test: {df_test.shape}, Core_ID: {df_test['Core_ID'].nunique()}")

    print("\nLabel Distribution:")
    print("\nTrain:")
    print(df_train["Label"].value_counts(normalize=True).sort_index())
    print("\nValidation:")
    print(df_val["Label"].value_counts(normalize=True).sort_index())
    print("\nTest:")
    print(df_test["Label"].value_counts(normalize=True).sort_index())

    os.makedirs("data_split", exist_ok=True)

    df_train.to_excel("data_split/train.xlsx", index=False)
    df_val.to_excel("data_split/val.xlsx", index=False)
    df_test.to_excel("data_split/test.xlsx", index=False)

    print("\nData saved:")
    print("data_split/train.xlsx")
    print("data_split/val.xlsx")
    print("data_split/test.xlsx")

    return df_train, df_val, df_test


if __name__ == "__main__":
    file_path = "data/training_dataset.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError("Please put training_dataset.xlsx in the data folder.")

    split_data(file_path)
