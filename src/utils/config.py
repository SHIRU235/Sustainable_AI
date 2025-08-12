import os

# ====== PATHS ======
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model paths
ENERGY_MODEL_PATH = os.path.join(MODEL_DIR, "energy_predictor", "energy_predictor.pkl")

# ====== DATA ======
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic")

# ====== APP SETTINGS ======
DEFAULT_RANDOM_STATE = 42
ENERGY_UNIT = "kWh"

# ====== LOGGING ======
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ====== MODEL PARAMETERS ======
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": DEFAULT_RANDOM_STATE,
}