from src.phy_conv.components.model_inferencing import InferenceEngine
from src.phy_conv import logger
from config.config import INFERENCE_DATA_CONFIG,MODEL_CONFIG

class InferencePipeline:
    def __init__(self):
        self.data_config = INFERENCE_DATA_CONFIG
        self.model_config = MODEL_CONFIG
        self.drop_cols = ['NPI_ID', 'HCP_ID']

    def run(self):
        logger.info("Starting inference pipeline...")

        engine = InferenceEngine(
            data_config=self.data_config,
            model_config=self.model_config,
            drop_columns=self.drop_cols
        )

        df = engine.load_data()
        model = engine.load_model()
        df_with_predictions = engine.predict()

        logger.info("Inference completed successfully.")
        return df_with_predictions
