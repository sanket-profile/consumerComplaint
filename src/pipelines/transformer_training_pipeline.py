from src.components.data_transformation_transformers import dataTransformation
from src.components.model_training_transformer import modelTraining
from src.components.data_ingestion import dataIngestion

from dataclasses import dataclass

class trainingPipeline():
    def __init__(self) -> None:
        pass
    
    def inititateTraining(self):
        data_ingestion = dataIngestion()
        df,path = data_ingestion.initiateDataIngestion()
        data_transformation = dataTransformation()
        train_dataset,test_dataset,_,_,_,_, = data_transformation.initiateDataTransformation(df=df)
        model_training = modelTraining()
        logisticPath,forestPath = model_training.initiateModelTraining(train_dataset=train_dataset,test_dataset=test_dataset)
        
        return (
            logisticPath,
            forestPath
        )
    
training_pipeline = trainingPipeline()
training_pipeline.inititateTraining()

training_pipeline = trainingPipeline()
training_pipeline.inititateTraining()