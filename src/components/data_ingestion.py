import os
import warnings
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException

@dataclass
class dataIngestionConfig():
    rawDataPath : str = os.path.join(os.getcwd(),"Artifacts","complaints.csv")

class dataIngestion():
    def __init__(self) -> None:
        self.data_Ingestion_Config = dataIngestionConfig()

    def initiateDataIngestion(self):
        try:
            logger.info("Starting Data Ingestion")

            df = pd.read_csv("/Users/sanketsaxena/Desktop/consumerComplaint/Artifacts/complaints.csv")

            logger.info("Data Ingestion completed")

            return (
                df,
                self.data_Ingestion_Config.rawDataPath
                )
        except Exception as e:
            raise CustomException("Something wrong in initiateDataIngestion method of dataIngestion class")
