import os
import pickle
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from src.phy_conv import logger

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target_column: str, model_params: dict,model_config:dict):
        self.df = df
        self.target_column = target_column
        self.model_params = model_params
        self.model_config = model_config
        self.model = None

    def split_data(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # First split 10% test
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        # Then split 20% of remaining as val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_pre, y_train_pre, test_size=0.2, random_state=42, stratify=y_train_pre
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train, y_train):
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
        logger.info('Model Trained successfully')
        return self.model
    
    def evaluate_model(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }
        return metrics
    
    def save_model(self):
        current_dir = os.getcwd()
        model_path = os.path.abspath(os.path.join(current_dir, self.model_config['output_dir'], self.model_config['filename']))
        logger.info(f'model path is: {model_path}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved at: {model_path}")
        return model_path
    def log_to_mlflow(self, metrics: dict):
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(self.model_config.get("mlflow_experiment_name", "default_experiment"))

            with mlflow.start_run(run_name=self.model_config.get("mlflow_run_name", "xgb_model_run")) as run:
                # Log parameters
                mlflow.log_params(self.model_params)

                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model
                 # Log the trained model and register it
                mlflow.xgboost.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="PhysicianConversionModel"
                )

                logger.info(f"Model logged and registered in MLflow. Run ID: {run.info.run_id}")