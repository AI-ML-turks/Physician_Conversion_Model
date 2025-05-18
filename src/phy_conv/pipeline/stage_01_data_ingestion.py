from src.phy_conv.components.data_ingestion import DataIngestion
from config.config import DATA_INGESTION_CONFIG


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        root_dir = DATA_INGESTION_CONFIG['root_dir']
        file_name = DATA_INGESTION_CONFIG['file_name']
        data_ingestion = DataIngestion(file_name,root_dir)
        df_input = data_ingestion.load_data()
        df_input.drop(['NPI_ID', 'HCP_ID'], axis=1, inplace = True, errors='ignore') 
        return df_input



    

