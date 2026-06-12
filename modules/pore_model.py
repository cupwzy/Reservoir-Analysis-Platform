import joblib
import os
import json

def load_model():

    config_path = "config/model_config.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError("Model config file not found")

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["current_model"]
    model_path = os.path.join("models", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)

def load_model_by_name(model_name):

    model_path = os.path.join("models", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)