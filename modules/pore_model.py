import joblib
import os

def load_model():

    model_path = "models/rf_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please train model locally and upload it to models/rf_model.pkl"
        )

    return joblib.load(model_path)