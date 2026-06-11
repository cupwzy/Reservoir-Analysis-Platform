import streamlit as st

# 页面配置
st.set_page_config(
    page_title="Reservoir Platform",
    page_icon="🛢️",
    layout="wide"
)

# =========================
# 界面风格
# =========================
st.markdown("""
<style>

/* ===== 全局背景 & 字体 ===== */
body {
    background-color: #0e1117;
    color: #e6edf3;
}

/* ===== 标题 ===== */
h1, h2, h3 {
    color: #f0f6fc;
}

/* ===== Sidebar 背景 ===== */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* ===== Sidebar 所有文字（关键修复）===== */
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
    font-weight: 500;
}

/* ===== 单选按钮布局优化 ===== */
div[role="radiogroup"] label {
    padding: 6px 10px;
    border-radius: 6px;
}

/* ===== 当前选中项（工业风高亮）===== */
div[role="radiogroup"] label[data-checked="true"] {
    background-color: #238636;
}

/* ===== 按钮 ===== */
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 6px;
    border: none;
}

/* ===== 文件上传 ===== */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    padding: 10px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)



# =========================
# 标题区域
# =========================
st.title("🛢️ Reservoir Analysis Platform")
st.caption("Integrated Digital Platform for Well Log Interpretation")

# =========================
# Sidebar（极简导航）
# =========================
st.sidebar.title("Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Upload", "Workspace"]
)

# =========================
# 全局数据
# =========================
if "data" not in st.session_state:
    st.session_state.data = None

# =========================
# HOME
# =========================
if menu == "Home":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Platform Overview")

        st.markdown("""
This platform is designed for digital reservoir workflows:

- Well log interpretation  
- Net pay calculation  
- Multi-zone analysis  
- AI-based pore typing  
- Visualization & reporting  

👉 Start from **Data Upload**
""")

    with col2:
        st.info("🧭 Workflow:\n1. Upload Data\n2. Process\n3. Analyze")

# =========================
# DATA UPLOAD
# =========================
elif menu == "Data Upload":
    st.subheader("Data Input")

    file = st.file_uploader(
        "Upload Well Log Data (Excel)",
        type=["xlsx"]
    )

    if file is not None:
        try:
            import pandas as pd

            df = pd.read_excel(file, engine="openpyxl")
            st.session_state.data = df

            st.success("✅ Data Loaded")

            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# WORKSPACE（未来所有功能入口）
# =========================
elif menu == "Workspace":
    st.subheader("Workspace")

    if st.session_state.data is None:
        st.warning("Please upload data first.")
    else:
        st.success("✅ Data Ready")

        # ✅ 这里是未来功能入口（非常关键）
        st.markdown("### Available Modules")

        st.markdown("""
- Interpretation (coming)  
- Zone Comparison (coming)  
- AI Pore Typing (coming)  
- Mapping (coming)  
""")

