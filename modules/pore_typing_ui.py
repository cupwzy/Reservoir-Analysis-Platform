import streamlit as st
import pandas as pd
from modules.micp_features import extract_micp_features
from modules.pore_model import load_model, predict_pore_type

def run():

    st.subheader("Pore Typing - MICP RF Model")

    uploaded_file = st.file_uploader("Upload MICP data", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file, skiprows=[1])

        feature_df = extract_micp_features(df)

        st.write("Extracted Features")
        st.dataframe(feature_df)

        model = load_model()

        pred, prob = predict_pore_type(model, feature_df)

        st.success(f"Pore Type: {pred[0]}")
        st.write("Confidence:", prob[0])