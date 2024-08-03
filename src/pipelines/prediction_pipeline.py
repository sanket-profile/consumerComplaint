import os
import pickle
import re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


from src.utils import change_product,merge_values,preprocess_text
from src.exception import CustomException
from src.logger import logger

class predictionPipeline():
    def __init__(self):
        pass

    def predict(self, X: str, tfidfPath: str, modelPath: str, lePath: str):
        try:
            logger.info("Initiating prediction")
            logger.info("Preprocessing the input text")

            X = preprocess_text(X)

            logger.info("Preprocessed the input text")
            logger.info("Loading the tfidf vectorizer and model")

            with open(tfidfPath, 'rb') as file:
                tfidf = pickle.load(file)

            with open(modelPath, 'rb') as file:
                model = pickle.load(file)

            with open(lePath, 'rb') as file:
                le = pickle.load(file)

            logger.info("Loaded the tfidf vectorizer and model")
            logger.info("Applying the tfidf vectorizer on out input text")

            X = tfidf.transform([X])

            logger.info("Applyied the tfidf vectorizer on out input text")
            logger.info("Initiating model and making it ready for prediction")

            prediction = model.predict(X)
            product = le.classes_[prediction]

            logger.info(f"prediction Completed. prediction is {product}")

            return product[0]


        except Exception as e:
            raise CustomException("Something wrong in predict method of predictionPipeline class")
        