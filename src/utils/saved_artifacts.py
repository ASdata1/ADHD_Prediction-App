import os
import json
import joblib



# --------------------------------------------------------
# Helper for directory creation
# --------------------------------------------------------
def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------
# MODEL SAVING & LOADING
# --------------------------------------------------------
def save_model(model, path):
    """
    Save a fitted ML model using joblib.
    """
    ensure_dir(os.path.dirname(path))

    try:
        joblib.dump(model, path)
        print(f"[OK] Saved model → {path}")
    except Exception as e:
        print(f"[ERROR] Could not save model to {path}: {e}")


def load_model(path):
    """
    Load a saved ML model.
    """
    try:
        model = joblib.load(path)
        print(f"[OK] Loaded model ← {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model from {path}: {e}")
        return None


# --------------------------------------------------------
# PREPROCESSOR SAVING (scaler, imputer, encoder, PCA)
# --------------------------------------------------------
def save_preprocessor(obj, name, folder):
    """
    Save any fitted preprocessor (scaler, encoder, etc.).
    """
    ensure_dir(folder)
    path = os.path.join(folder, f"{name}.joblib")

    try:
        joblib.dump(obj, path)
        print(f"[OK] Saved preprocessor {name} → {path}")
    except Exception as e:
        print(f"[ERROR] Could not save {name}: {e}")


def load_preprocessor(name, folder):
    """
    Load a specific fitted preprocessor.
    """
    path = os.path.join(folder, f"{name}.joblib")

    try:
        obj = joblib.load(path)
        print(f"[OK] Loaded preprocessor {name} ← {path}")
        return obj
    except Exception as e:
        print(f"[ERROR] Could not load {name}: {e}")
        return None


# --------------------------------------------------------
# METRICS SAVING
# --------------------------------------------------------
def save_metrics(metrics_dict, path):
    """
    Save evaluation metrics as a JSON file.
    """
    ensure_dir(os.path.dirname(path))

    try:
        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"[OK] Saved metrics → {path}")
    except Exception as e:
        print(f"[ERROR] Could not save metrics to {path}: {e}")


def load_metrics(path):
    """
    Load metrics JSON file.
    """
    try:
        with open(path, "r") as f:
            metrics = json.load(f)
        print(f"[OK] Loaded metrics ← {path}")
        return metrics
    except Exception as e:
        print(f"[ERROR] Could not load metrics from {path}: {e}")
        return None


# --------------------------------------------------------
# HYPERPARAMETER SAVING
# --------------------------------------------------------
def save_params(params_dict, path):
    """
    Save tuned or baseline parameters to JSON.
    """
    ensure_dir(os.path.dirname(path))

    try:
        with open(path, "w") as f:
            json.dump(params_dict, f, indent=4)
        print(f"[OK] Saved parameters → {path}")
    except Exception as e:
        print(f"[ERROR] Could not save parameters: {e}")


def load_params(path):
    """
    Load parameters JSON.
    """
    try:
        with open(path, "r") as f:
            params = json.load(f)
        print(f"[OK] Loaded parameters ← {path}")
        return params
    except Exception as e:
        print(f"[ERROR] Could not load parameters: {e}")
        return None


# --------------------------------------------------------
# PREDICTION SAVING
# --------------------------------------------------------
def save_predictions(preds, path):
    """
    Save model predictions as a simple list to JSON.
    """
    ensure_dir(os.path.dirname(path))

    try:
        with open(path, "w") as f:
            json.dump(list(preds), f, indent=4)
        print(f"[OK] Saved predictions → {path}")
    except Exception as e:
        print(f"[ERROR] Could not save predictions: {e}")


def load_predictions(path):
    """
    Load predictions from JSON.
    """
    try:
        with open(path, "r") as f:
            preds = json.load(f)
        print(f"[OK] Loaded predictions ← {path}")
        return preds
    except Exception as e:
        print(f"[ERROR] Could not load predictions: {e}")
        return None


# --------------------------------------------------------
# TIMESTAMPED SAVING (OPTIONAL, USEFUL)
# --------------------------------------------------------

