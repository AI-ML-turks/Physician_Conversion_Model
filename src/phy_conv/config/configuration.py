from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
import os


class ConfigurationManager:
#reading the YAML file
#creating root directories
#getting the data ingestion config from yaml file
    def __init__(self,config_file_path = CONFIG_FILE_PATH,params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path) #returns configbox objects
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root]) #create artifacts root directory
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )

        return data_ingestion_config