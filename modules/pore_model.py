import joblib
import os

def load_model():

    model_path = "models/rf_model.pkl"

    if not os.path.exists(model_path):

        print("Model not found. Training model...")

        from train_model import train_and_save_model
        train_and_save_model()

    return joblib.load(model_path)
