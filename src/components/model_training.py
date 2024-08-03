import os
import mlflow
import pickle

import warnings
warnings.simplefilter("ignore")


import dagshub
dagshub.init(repo_owner='sanket-profile', repo_name='consumerComplaint', mlflow=True)

import numpy as np

from src.exception import CustomException
from src.logger import logger


from mlflow.sklearn import log_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

from dataclasses import dataclass

@dataclass

class modelTrainingConfig():
    logisticModelPath : str = os.path.join(os.getcwd(),"Artifacts","logisticModel.pkl")
    forestModelPath : str = os.path.join(os.getcwd(),"Artifacts","forestModel.pkl")

class modelTraining():
    def __init__(self) -> None:
        self.model_training_config = modelTrainingConfig()

    def initiateModelTraining(self,X_train,X_test,y_train,y_test):
        try:
            mlflow.set_tracking_uri(
                    "https://dagshub.com/sanket-profile/consumerComplaint.mlflow"
                    )
            mlflow.set_experiment("Testing Complaints")
            with mlflow.start_run():
                logger.info("Starting the training process")
                logger.info("Loading and training Logistic Regression and RandomForestClassifier")

                models = [
                    RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),
                    LogisticRegression(random_state=42)
                ]
                i = 0
                logger.info("I am here")
                for model in models:
                    logger.info("I am here")
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    
                    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
                    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
                    mlflow.sklearn.log_model(model, f"{model.__class__.__name__}",signature=signature)
                    
                
                    if i == 0:
                        mlflow.log_params({"n_estimators": 10, "max_depth": 5})

                    report = classification_report(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)

                    logger.info(f"Model: {model.__class__.__name__}")
                    logger.info("Accuracy:\n%s", accuracy)
                    logger.info("Classification Report:\n%s", report)

                    logger.info(f"Saving {model.__class__.__name__} into pickle file in Artifacts Folder")

                    if i == 0:
                        pickle_file = self.model_training_config.forestModelPath
                    else:
                        pickle_file = self.model_training_config.logisticModelPath

                    with open(pickle_file, 'wb') as file:
                        pickle.dump(model, file)

                    logger.info(f"Saved {model.__class__.__name__} model into pickle file")

                return (
                self.model_training_config.logisticModelPath,
                self.model_training_config.forestModelPath
                )

        except Exception as e:
            raise CustomException("Something wrong in initiateModelTraining method of modelTraining class")
        