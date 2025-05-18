import os
import pandas as pd

class DataIngestion:
    def __init__(self, file_name, data_dir):
        self.current_dir = os.getcwd()
        self.file_path = os.path.abspath(os.path.join(self.current_dir, data_dir, file_name))
        self.df_input = None

    def load_data(self):
        try:
            self.df_input = pd.read_csv(self.file_path)
            self.clean_columns()
            print("Data loaded successfully.")
            print("Shape and data format:")
            print(self.df_input.shape)
            return self.df_input
        except FileNotFoundError:
            print(f"File not found at: {self.file_path}")
            raise
        except Exception as e:
            print("An error occurred while loading the data.")
            raise e

    def clean_columns(self):
        self.df_input.columns = self.df_input.columns.str.strip().str.replace(' ', '_')
