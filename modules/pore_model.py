import joblib

def load_model():
    return joblib.load("models/rf_model.pkl")


def predict_pore_type(model, feature_df):
    pred = model.predict(feature_df)
    prob = model.predict_proba(feature_df)

    return pred, prob