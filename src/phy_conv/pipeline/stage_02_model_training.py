import pandas as pd
from src.phy_conv.components.model_training import ModelTrainer
from config.config import MODEL_HYPER_PARAMS, TARGET_COLUMN, MODEL_TRAINING_CONFIG
from src.phy_conv import logger


class ModelTrainingPipeline:
    def __init__(self, input_df: pd.DataFrame):
        self.input_df = input_df
        self.target_column = TARGET_COLUMN
        self.model_params = MODEL_HYPER_PARAMS
        self.model_config = MODEL_TRAINING_CONFIG

    def train(self):
        logger.info("Training pipeline started")
        trainer = ModelTrainer(
            df=self.input_df,
            target_column=self.target_column,
            model_params=self.model_params,
            model_config=self.model_config
        )

        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data()
        logger.info("Data split completed")

        model = trainer.train_model(X_train, y_train)
        model_path = trainer.save_model()

        metrics = trainer.evaluate_model(model, X_val, y_val)
        logger.info(f"Validation metrics: {metrics}")

        trainer.log_to_mlflow(metrics)

        logger.info("Model trained, evaluated, and logged successfully.")
        return model, model_path, (X_train, X_val, X_test, y_train, y_val, y_test)
