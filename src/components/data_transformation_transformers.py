import os
import sys
import pickle

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words_list = stopwords.words("english")

import pandas as pd
import numpy as np
import imblearn
from scipy.sparse import save_npz

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.utils import change_product,merge_values,preprocess_text

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from transformers import DistilBertTokenizer

@dataclass
class dataTransformationTransformerConfig():
    Xtrain_encoding_tokenized : str = os.path.join(os.getcwd(),"Artifacts","train_encodings_tokenized.pkl")
    Xtest_encoding_tokenized : str = os.path.join(os.getcwd(),"Artifacts","train_encodings_tokenized.pkl")
    ytrain_encoded_path : str = os.path.join(os.getcwd(),"Artifacts","y_train_encoded_transformers.npy")
    ytest_encoded_path : str = os.path.join(os.getcwd(),"Artifacts","y_test_encoded_transformers.npy")
    train_dataset_transformers : str = os.path.join(os.getcwd(),"Artifacts","tensorflowDatasetTrain")
    test_dataset_transformers : str = os.path.join(os.getcwd(),"Artifacts","tensorflowDatasetTest")

class dataTransformation():
    def __init__(self) -> None:
        self.data_transformation_config = dataTransformationTransformerConfig()

    def initiateDataTransformation(self,df: pd.DataFrame):
        try:
            logger.info("Starting data transformation")
            logger.info("Removing Unwanted Columns")

            df.drop(['Date received','Company public response', 'Company','State', 'ZIP code', 'Tags', 'Consumer consent provided?','Submitted via', 'Date sent to company', 'Company response to consumer','Timely response?', 'Consumer disputed?', 'Complaint ID'],axis = 1,inplace = True)

            logger.info("Removed Unwanted Columns")
            logger.info("Changing Product name where Other financial service was not needed to the real Product that should be there")

            mask = (df['Product'] == "Other financial service") & (df['Sub-product'] == "Credit repair")
            df.loc[mask, 'Product'] = df.loc[mask, 'Product'].apply(change_product)

            mask = (df['Product'] == "Other financial service") & (df['Sub-product'] == "Debt settlement")
            df.loc[mask, 'Product'] = df.loc[mask, 'Product'].apply(change_product)

            mask = (df['Product'] == "Other financial service") & (df['Sub-product'] == "Money order")
            df.loc[mask, 'Product'] = df.loc[mask, 'Product'].apply(change_product)

            logger.info("Changed Product name where Other financial service was not needed to the real Product that should be there")
            logger.info("Merging similar products to simplify target variable")

            df['Product'] = df['Product'].apply(lambda x : merge_values(x,["Credit reporting, credit repair services, or other personal consumer reports","Credit reporting or other personal consumer reports","Credit reporting","Credit Reporting"],"Credit Reporting/Repair Services/Consumer Reports"))
            df['Product'] = df['Product'].apply(lambda x : merge_values(x,["Credit card or prepaid card","Credit card","Prepaid card"],"Credit/Prepaid Cards"))
            df['Product'] = df['Product'].apply(lambda x : merge_values(x,["Payday loan, title loan, or personal loan","Payday loan, title loan, personal loan, or advance loan","Payday loan","Consumer Loan"],"Personal Loan"))
            df['Product'] = df['Product'].apply(lambda x : merge_values(x,["Money transfer, virtual currency, or money service","Money transfers","Virtual currency"],"Money Services/Transfer"))
            df['Product'] = df['Product'].apply(lambda x : merge_values(x,["Bank account or service","Checking or savings account"],"Bank Account Services"))
            df['Product'] = df['Product'].apply(lambda x : merge_values(x,["Debt collection","Debt or credit management"],"Debt Collection"))

            logger.info("Merged similar products to simplify target variable")
            logger.info("Changing the subProduct value from I do not know to -> Debt Collection so that it could be usefull for the customer complaint column")

            df['Sub-product'] = df['Sub-product'].apply(lambda x : merge_values(x,["I do not know"],"Debt Collection"))

            logger.info("Changed the subProduct value from I do not know to -> Debt Collection")
            logger.info("Replacing all NaN values with the empty strings so that we can combine all the columns into one")

            df['Sub-product'] = df['Sub-product'].fillna('')
            df['Issue'] = df['Issue'].fillna('')
            df['Sub-issue'] = df['Sub-issue'].fillna('')
            df['Consumer complaint narrative'] = df['Consumer complaint narrative'].fillna('')

            logger.info("Replaced all NaN values with the empty strings")
            logger.info("Combining all the columns to form single input column")

            df['Consumer Complaint'] = df['Sub-product'] + " " + df['Issue'] + " " + df['Sub-issue'] + " " + df['Consumer complaint narrative']

            logger.info("Combined all the columns to form single input column")
            logger.info("Removing for duplicate rows")

            df_cleaned = df.drop_duplicates(keep='first')

            logger.info("Removed for duplicate rows")
            logger.info("Downsampling the train data,since it has imbalance in classes. Also creating train test split")

            X = df_cleaned['Consumer Complaint']
            y = df_cleaned['Product']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            
            train_data = pd.DataFrame({'text': X_train, 'label': y_train})
            desired_samples_train = {
                'Credit Reporting/Repair Services/Consumer Reports': 30000,
                'Debt Collection': len(train_data[train_data['label'] == 'Debt Collection'])//30,
                'Credit/Prepaid Cards': len(train_data[train_data['label'] == 'Credit/Prepaid Cards'])//30,
                'Mortgage': len(train_data[train_data['label'] == 'Mortgage'])//30,
                'Bank Account Services': len(train_data[train_data['label'] == 'Bank Account Services'])//30,
                'Student loan': len(train_data[train_data['label'] == 'Student loan'])//30,
                'Money Services/Transfer': len(train_data[train_data['label'] == 'Money Services/Transfer'])//30,
                'Personal Loan': len(train_data[train_data['label'] == 'Personal Loan'])//30,
                'Vehicle loan or lease': len(train_data[train_data['label'] == 'Vehicle loan or lease'])//30,
                'Other financial service': len(train_data[train_data['label'] == 'Other financial service'])//30  # Keep 105 examples for this class
            }
            rus = RandomUnderSampler(sampling_strategy = desired_samples_train,random_state=42)
            X_train_res, y_train_resampled = rus.fit_resample(train_data[['text']], train_data['label'])

            logger.info("Downsampled the train data. Also created train test split")
            logger.info("Downsampling the test data,since it has imbalance in classes.")

            test_data = pd.DataFrame({'text': X_test, 'label': y_test})
            desired_samples_test = {
                'Credit Reporting/Repair Services/Consumer Reports': 3230,
                'Debt Collection': len(test_data[test_data['label'] == 'Debt Collection'])//30,
                'Credit/Prepaid Cards': len(test_data[test_data['label'] == 'Credit/Prepaid Cards'])//30,
                'Mortgage': len(test_data[test_data['label'] == 'Mortgage'])//30,
                'Bank Account Services': len(test_data[test_data['label'] == 'Bank Account Services'])//30,
                'Student loan': len(test_data[test_data['label'] == 'Student loan'])//30,
                'Money Services/Transfer': len(test_data[test_data['label'] == 'Money Services/Transfer'])//30,
                'Personal Loan': len(test_data[test_data['label'] == 'Personal Loan'])//30,
                'Vehicle loan or lease': len(test_data[test_data['label'] == 'Vehicle loan or lease'])//30,
                'Other financial service': len(test_data[test_data['label'] == 'Other financial service'])//30  # Keep 105 examples for this class
            }
            rus = RandomUnderSampler(sampling_strategy = desired_samples_test,random_state=42)
            X_test_res, y_test_resampled = rus.fit_resample(test_data[['text']], test_data['label'])

            logger.info("Downsampled the test data,since it has imbalance in classes.")
            logger.info("Preprocessing the train and text data")
            X_train_resampled = X_train_res
            X_test_resampled = X_test_res
            X_train_resampled['text'] = X_train_res['text'].apply(preprocess_text)
            X_test_resampled['text'] = X_test_res['text'].apply(preprocess_text)

            logger.info(f"Tokenizing the downsampled X_train and X_test {X_train_resampled.columns} {X_test_resampled.columns}")

            model_name = 'distilbert-base-uncased'
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            train_encodings= tokenizer(X_train_resampled['text'].tolist(), return_tensors='tf', max_length=512, padding='max_length', truncation=True)
            test_encodings= tokenizer(X_test_resampled['text'].tolist(), return_tensors='tf', max_length=512, padding='max_length', truncation=True)

            logger.info("Tokenized the downsampled X_train and X_test")
            logger.info("Label Encoding target columns")

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_resampled.values)
            y_test_encoded = le.transform(y_test_resampled.values)

            logger.info("Completed Label Encoding target columns")
            logger.info("Converting training encodings and test encoding to tensorflow dataset type to feed to transformers")

            train_dataset = tf.data.Dataset.from_tensor_slices(({
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask']
            }, y_train_encoded))

            test_dataset = tf.data.Dataset.from_tensor_slices(({
                'input_ids': test_encodings['input_ids'],
                'attention_mask': test_encodings['attention_mask']
            }, y_test_encoded))   

            logger.info("Saving train and test encodings and train and test dataset into Artifacts Folder") 

            with open(self.data_transformation_config.Xtrain_encoding_tokenized, 'wb') as f:
                pickle.dump(train_encodings, f)

            with open(self.data_transformation_config.Xtest_encoding_tokenized, 'wb') as f:
                pickle.dump(test_encodings, f)

            tf.data.experimental.save(train_dataset, self.data_transformation_config.train_dataset_transformers)
            tf.data.experimental.save(test_dataset, self.data_transformation_config.test_dataset_transformers)

            np.save(self.data_transformation_config.ytrain_encoded_path,y_train_encoded)
            np.save(self.data_transformation_config.ytest_encoded_path,y_test_encoded)

            logger.info("Saved train and test encodings and train and test dataset into Artifacts Folder")

            return (
                self.data_transformation_config.train_dataset_transformers,
                self.data_transformation_config.test_dataset_transformers,
                self.data_transformation_config.Xtrain_encoding_tokenized,
                self.data_transformation_config.Xtest_encoding_tokenized,
                self.data_transformation_config.ytrain_encoded_path,
                self.data_transformation_config.ytest_encoded_path
            )
                  

        except Exception as e:
            raise CustomException("Something wrong in initiateDataTransformation method of dataTransformation class")