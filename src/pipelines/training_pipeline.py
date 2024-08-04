from src.components.data_transformation import dataTransformation
from src.components.model_training import modelTraining
from src.components.data_ingestion import dataIngestion

from dataclasses import dataclass

class trainingPipeline():
    def __init__(self) -> None:
        pass
    
    def inititateTraining(self):
        data_ingestion = dataIngestion()
        df,path = data_ingestion.initiateDataIngestion()
        data_transformation = dataTransformation()
        X_train_tfidf,X_test_tfidf,y_train,y_test,_,_,_,_,tfidf,le = data_transformation.initiateDataTransformation(df=df)
        model_training = modelTraining()
        logisticPath,forestPath = model_training.initiateModelTraining(X_train=X_train_tfidf,X_test=X_test_tfidf,y_train=y_train,y_test=y_test)
        
        return (
            logisticPath,
            forestPath
        )
    
training_pipeline = trainingPipeline()
training_pipeline.inititateTraining()