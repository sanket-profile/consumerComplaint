import os
import mlflow
import pickle

import warnings
warnings.simplefilter("ignore")


import dagshub
dagshub.init(repo_owner='sanket-profile', repo_name='consumerComplaint', mlflow=True)

import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

from src.exception import CustomException
from src.logger import logger


from mlflow.sklearn import log_model
from sklearn.metrics import accuracy_score, classification_report

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

from dataclasses import dataclass

@dataclass

class modelTrainingTransformerConfig():
    ModelPath : str = os.path.join(os.getcwd(),"Artifacts","transformerModel.h5")

class modelTraining():
    def __init__(self) -> None:
        self.model_training_config = modelTrainingTransformerConfig()

    def initiateModelTraining(self, train_dataset, test_dataset):
        try:
            mlflow.set_tracking_uri(
                    "https://dagshub.com/sanket-profile/consumerComplaint.mlflow"
                    )
            mlflow.set_experiment("Testing Complaints")
            with mlflow.start_run():
                logger.info("Starting the training process")
                logger.info("Loading Train and Test DataSet")
                
                element_spec = (
                    {
                        'input_ids': tf.TensorSpec(shape=(512,), dtype=tf.int32),
                        'attention_mask': tf.TensorSpec(shape=(512,), dtype=tf.int32)
                    },
                    tf.TensorSpec(shape=(), dtype=tf.int64)
                )

                loaded_train_dataset = tf.data.experimental.load(train_dataset, element_spec=element_spec)
                loaded_test_dataset = tf.data.experimental.load(test_dataset, element_spec=element_spec)

                logger.info("Loaded Train and Test DataSet")
                logger.info("Loading and training the model")

                train_dataset = loaded_train_dataset.shuffle(10).batch(32)
                test_dataset = loaded_test_dataset.batch(32)

                model_name = 'distilbert-base-uncased'
                model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)

                optimizer = tf.keras.optimizers.legacy.Adam()
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                model.fit(train_dataset, epochs=2, validation_data=test_dataset, verbose=1)

                logger.info("Model trained")
                logger.info("Saving the model in h5 format")

                model.save(self.model_training_config.ModelPath, save_format='h5')

                logger.info(f"Saving model in Artifacts Folder")

                return (
                self.model_training_config.ModelPath
                )

        except Exception as e:
            raise CustomException("Something wrong in initiateModelTraining method of modelTraining class")
        