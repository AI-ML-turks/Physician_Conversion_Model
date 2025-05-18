DATA_INGESTION_CONFIG = {
    "root_dir":"data/final_data", "file_name":"model_input.csv"
}


MODEL_HYPER_PARAMS = {
    'colsample_bytree': 0.8011,
    'gamma': 0.00033,
    'max_depth': 7,
    'reg_alpha': 0.2006,
    'subsample': 0.1926
}

MODEL_TRAINING_CONFIG = {
    "filename":"phy_conv.pkl",  "output_dir":"model","mlflow_experiment_name": "Physician Conversion Classifier Experiments",
    "mlflow_run_name": "xgboost_pipeline_run_2"
}

TARGET_COLUMN = "TARGET"


# Config dictionaries for modularity
INFERENCE_DATA_CONFIG = {
    "data_dir": "data/Inference_data",
    "filename": "Inference_data.csv"
}

MODEL_CONFIG = {
    "model_dir": "model",
    "filename": "physician_conversion.pkl"
}