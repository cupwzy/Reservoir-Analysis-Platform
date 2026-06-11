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

/* 全局 */
body {
    background-color: #0e1117;
    color: #e6edf3;
}

/* 标题 */
h1, h2, h3 {
    color: #f0f6fc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
    font-weight: 500;
}

/* 单选按钮 */
div[role="radiogroup"] label {
    padding: 6px 10px;
    border-radius: 6px;
}

div[role="radiogroup"] label[data-checked="true"] {
    background-color: #238636;
}

/* 按钮 */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 6px;
    border: none;
}

/* 上传框 */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    padding: 10px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 标题
# =========================
st.title("Reservoir Analysis Platform")
st.caption("Integrated Digital Platform for Well Log Interpretation")

# =========================
# Sidebar
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
# Home
# =========================
if menu == "Home":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Platform Overview")

        st.markdown("""
This platform supports:

- Well log interpretation  
- Net pay calculation  
- Multi-zone analysis  
- AI-based pore typing  
- Visualization and reporting  

Start from Data Upload.
""")

    with col2:
        st.info("Workflow:\n1. Upload Data\n2. Process\n3. Analyze")

# =========================
# Data Upload
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

            st.success("Data loaded successfully")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# Workspace
# =========================
elif menu == "Workspace":
    st.subheader("Workspace")

    if st.session_state.data is None:
        st.warning("Please upload data first.")
    else:
        st.success("Data ready")

        st.markdown("### Available Modules")

        st.markdown("""
- Interpretation (coming)
- Zone Comparison (coming)
- AI Pore Typing (coming)
- Mapping (coming)
""")