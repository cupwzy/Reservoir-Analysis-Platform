import pandas as pd


# =========================
# 主函数：读取 + 合并
# =========================
def load_and_merge_well_data(files):
    """
    Load multiple well interpretation Excel files,
    clean invalid rows, and merge into a single DataFrame.

    Parameters
    ----------
    files : list
        Uploaded files from Streamlit

    Returns
    -------
    DataFrame
        Combined well dataset
    """

    all_data = []

    for file in files:
        df = pd.read_excel(file, engine="openpyxl")

        # =========================
        # 获取井名
        # =========================
        if "Well" in df.columns:
            well_name = str(df["Well"].iloc[0])
        else:
            well_name = file.name.split(".")[0]

        df = df.copy()
        df["Well"] = well_name

        # =========================
        # 数据清洗（关键！！）
        # =========================

        # 删除空 ZONE
        if "ZONE" in df.columns:
            df = df[df["ZONE"].notna()]

        # 删除 Cutoffs/说明文字（你遇到的问题）
        df = df[~df["ZONE"].astype(str).str.contains(":", na=False)]

        # 删除异常长字符串（说明类文本）
        df = df[df["ZONE"].astype(str).str.len() < 20]

        # =========================
        # 类型转换（防止 TypeError）
        # =========================
        numeric_cols = ["MD_THK", "TVDSS_THK", "VSH", "PHIE", "SWE"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 删除无效数据行（厚度为空）
        if "TVDSS_THK" in df.columns:
            df = df[df["TVDSS_THK"].notna()]

        # =========================
        # 可选：删除全部为空行
        # =========================
        df = df.dropna(how="all")

        all_data.append(df)

    # =========================
    # 合并所有井
    # =========================
    if len(all_data) == 0:
        raise ValueError("No valid data found in uploaded files.")

    df_all = pd.concat(all_data, ignore_index=True)

    return df_all


# =========================
# 可选扩展：字段校验
# =========================
def validate_well_data(df):
    required_cols = ["ZONE", "TVDSS_THK"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
