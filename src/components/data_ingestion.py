import os
import pandas as pd
import numpy as np

from scipy.sparse import load_npz

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.components.data_transformation import dataTransformation
from src.components.model_training import modelTraining

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


if __name__ == "__main__":
    """data_ingestion = dataIngestion()
    df,path = data_ingestion.initiateDataIngestion()
    data_transformation = dataTransformation()
    data_transformation.initiateDataTransformation(df=df)"""


    X_train_tfidf = load_npz("/Users/sanketsaxena/Desktop/consumerComplaint/Artifacts/Xtrain_tfidf.npz")
    X_test_tfidf = load_npz("/Users/sanketsaxena/Desktop/consumerComplaint/Artifacts/Xtest_tfidf.npz")
    y_train = np.load('/Users/sanketsaxena/Desktop/consumerComplaint/Artifacts/y_train_encoded.npy')
    y_test = np.load('/Users/sanketsaxena/Desktop/consumerComplaint/Artifacts/y_test_encoded.npy')
    model_training = modelTraining()
    model_training.initiateModelTraining(X_train=X_train_tfidf,X_test=X_test_tfidf,y_train=y_train,y_test=y_test)