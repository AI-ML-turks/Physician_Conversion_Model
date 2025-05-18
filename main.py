from src.phy_conv.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.phy_conv.pipeline.stage_02_model_training import ModelTrainingPipeline
from src.phy_conv.pipeline.stage_03_model_inferencing import InferencePipeline
from src.phy_conv import logger

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"{STAGE_NAME} started")
    data_ingestion = DataIngestionTrainingPipeline()
    df_input = data_ingestion.main()
    logger.info(f"{STAGE_NAME} finished")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training & Evaluation"

try:
    logger.info(f"{STAGE_NAME} started")
    model_training = ModelTrainingPipeline(df_input)
    model_training.train()
    logger.info(f"{STAGE_NAME} finished")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Inferencing"

try:
    logger.info(f"{STAGE_NAME} started")
    inference_pipeline = InferencePipeline()
    df_inference = inference_pipeline.run()
    logger.info(f"value_counts for pred: {df_inference['Prediction'].value_counts()}")
    logger.info(f"{STAGE_NAME} finished")
    
except Exception as e:
    logger.exception(e)
    raise e
