import os
import pandas as pd
import numpy as np

from scipy.sparse import load_npz

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.components.data_transformation import dataTransformation
from src.components.model_training import modelTraining
from src.pipelines.prediction_pipeline import predictionPipeline

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
    
    data_ingestion = dataIngestion()
    df,path = data_ingestion.initiateDataIngestion()

    data_transformation = dataTransformation()
    X_train_tfidf,X_test_tfidf,y_train,y_test,_,_,_,_,tfidf,le = data_transformation.initiateDataTransformation(df=df)

    model_training = modelTraining()
    logisticPath,forestPath = model_training.initiateModelTraining(X_train=X_train_tfidf,X_test=X_test_tfidf,y_train=y_train,y_test=y_test)
    
    x ="credit reporting incorrect information on your report information belongs to someone else this is my numerous request that i have been a victim of identity theft and that no one seems to care, that i want to dispute specific records in my credit file that do not belong to me, or that i have signed any agreement. the accounts i'm challenging have nothing to do with any transactions i've done or authorized to gain products, services, or money. please remove the following ; xxxx xxxx ; xxxx xxxx ; xxxx xxxx xxxx xxxx xxxx xxxx, tx xxxx xxxx ; xxxx xxxx xxxx xxxx xxxx, tx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx, tx xxxx xxxx xxxx xxxx xxxx xxxx xxxx, tx xxxx xxxx xxxx  xxxx balance : {$ .  } ; xxxx xxxx balance : {$ .  } ; xxxx xxxx xxxx balance : {$ .  } ; xxxx xxxx balance : {$ .  } ; xxxxxxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxxxxxx xxxx balance : {$ .  } ; xxxx xxxxxxxx balance : {$ .  } ; xxxx  xxxx balance : {$ .  } xxxx xxxx xxxx xxxx reference # : xxxx date filed/reported : xx/xx/xxxx ; xxxx xxxx xx/xx/xxxx ; us sm bus admin oda xx/xx/xxxx ; xxxx - xx/xx/xxxx ; xxxx xxxx xxxx xxxx xxxx  xxxx xx/xx/xxxx ; xxxx xxxx ; xxxx  xx/xx/xxxx xxxx xxxx xx/xx/xxxx xxxx xxxx xx/xx/xxxx ; xxxx xx/xx/xxxx"
   
    prediction_pipeline = predictionPipeline()
    print(prediction_pipeline.predict(X=x,tfidfPath=tfidf,modelPath=logisticPath,lePath=le))