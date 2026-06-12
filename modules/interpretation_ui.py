import streamlit as st

def zone_viewer(well_dict):
    st.subheader("ZONE Analysis")

    # 选择井
    wells = list(well_dict.keys())
    selected_well = st.selectbox("Select Well", wells)

    df = well_dict[selected_well]

    # 获取ZONE列表
    zones = df["ZONE"].dropna().unique()

    selected_zones = st.multiselect(
        "Select ZONE",
        options=zones,
        default=zones
    )

    # 筛选
    df_filtered = df[df["ZONE"].isin(selected_zones)]

    # 显示结果
    st.write("Filtered Data")
    st.dataframe(df_filtered, use_container_width=True)

    # 简单统计（可扩展）
    st.markdown("### Summary")

    summary = df_filtered.groupby("ZONE").agg({
        "TVDSS_THK": "sum",
        "PHIE": "mean",
        "SWE": "mean",
        "VSH": "mean"
    }).reset_index()

    st.dataframe(summary, use_container_width=True)