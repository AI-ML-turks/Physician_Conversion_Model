import os
import pickle
import pandas as pd
from src.phy_conv import logger

class InferenceEngine:
    def __init__(self, data_config: dict, model_config: dict, drop_columns: list):
        self.data_config = data_config
        self.model_config = model_config
        self.drop_columns = drop_columns
        self.df = None
        self.model = None

    def load_data(self):
        current_dir = os.getcwd()
        file_path = os.path.abspath(os.path.join(current_dir, self.data_config['data_dir'], self.data_config['filename']))
        logger.info(f"Loading inference data from: {file_path}")

        self.df = pd.read_csv(file_path)
        if 'Unnamed: 0' in self.df.columns:
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        logger.info(f"shape of data is : {self.df.shape}")
        return self.df

    def load_model(self):
        current_dir = os.getcwd()
        model_path = os.path.abspath(os.path.join(current_dir, self.model_config['model_dir'], self.model_config['filename']))
        logger.info(f"Loading model from: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        return self.model

    def predict(self):
        if self.df is None or self.model is None:
            raise Exception("Data or Model not loaded.")
        logger.info("Performing inference...")
        self.df['Prediction'] = self.model.predict(self.df.drop(self.drop_columns, axis=1, errors='ignore'))
        return self.df
