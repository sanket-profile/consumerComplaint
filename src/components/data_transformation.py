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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


@dataclass
class dataTransformationConfig():
    Xtrain_tfidf_path : str = os.path.join(os.getcwd(),"Artifacts","Xtrain_tfidf.npz")
    Xtest_tfidf_path : str = os.path.join(os.getcwd(),"Artifacts","Xtest_tfidf.npz")
    ytrain_encoded_path : str = os.path.join(os.getcwd(),"Artifacts","y_train_encoded.npy")
    ytest_encoded_path : str = os.path.join(os.getcwd(),"Artifacts","y_test_encoded.npy")
    tfidf_path : str = os.path.join(os.getcwd(),"Artifacts","tfidf.pkl")
    lableEncoder_path : str = os.path.join(os.getcwd(),"Artifacts","le.pkl")

class dataTransformation():
    def __init__(self) -> None:
        self.data_transformation_config = dataTransformationConfig()

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
                'Credit Reporting/Repair Services/Consumer Reports': 300000,
                'Debt Collection': len(train_data[train_data['label'] == 'Debt Collection']),
                'Credit/Prepaid Cards': len(train_data[train_data['label'] == 'Credit/Prepaid Cards']),
                'Mortgage': len(train_data[train_data['label'] == 'Mortgage']),
                'Bank Account Services': len(train_data[train_data['label'] == 'Bank Account Services']),
                'Student loan': len(train_data[train_data['label'] == 'Student loan']),
                'Money Services/Transfer': len(train_data[train_data['label'] == 'Money Services/Transfer']),
                'Personal Loan': len(train_data[train_data['label'] == 'Personal Loan']),
                'Vehicle loan or lease': len(train_data[train_data['label'] == 'Vehicle loan or lease']),
                'Other financial service': len(train_data[train_data['label'] == 'Other financial service'])  # Keep 105 examples for this class
            }
            rus = RandomUnderSampler(sampling_strategy = desired_samples_train,random_state=42)
            X_train_res, y_train_res = rus.fit_resample(train_data[['text']], train_data['label'])
            X_train_res = X_train_res['text']

            logger.info("Downsampled the train data. Also created train test split")
            logger.info("Downsampling the test data,since it has imbalance in classes.")

            test_data = pd.DataFrame({'text': X_test, 'label': y_test})
            desired_samples_test = {
                'Credit Reporting/Repair Services/Consumer Reports': 32302,
                'Debt Collection': len(test_data[test_data['label'] == 'Debt Collection']),
                'Credit/Prepaid Cards': len(test_data[test_data['label'] == 'Credit/Prepaid Cards']),
                'Mortgage': len(test_data[test_data['label'] == 'Mortgage']),
                'Bank Account Services': len(test_data[test_data['label'] == 'Bank Account Services']),
                'Student loan': len(test_data[test_data['label'] == 'Student loan']),
                'Money Services/Transfer': len(test_data[test_data['label'] == 'Money Services/Transfer']),
                'Personal Loan': len(test_data[test_data['label'] == 'Personal Loan']),
                'Vehicle loan or lease': len(test_data[test_data['label'] == 'Vehicle loan or lease']),
                'Other financial service': len(test_data[test_data['label'] == 'Other financial service'])  # Keep 105 examples for this class
            }
            rus = RandomUnderSampler(sampling_strategy = desired_samples_test,random_state=42)
            X_test_res, y_test_res = rus.fit_resample(test_data[['text']], test_data['label'])
            X_test_res = X_test_res['text']

            logger.info("Downsampled the test data,since it has imbalance in classes.")
            logger.info("Preprocessing the train and text data")

            X_train_res= X_train_res.apply(preprocess_text)
            X_test_res= X_test_res.apply(preprocess_text)

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        stop_words='english')
            
            X_train_tfidf = tfidf.fit_transform(X_train_res)
            X_test_tfidf = tfidf.transform(X_test_res)

            logger.info("Converted text data into numerical vector representation using TF-IDF")
            logger.info("Converting ylabels to classes using LabelEncoder")

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_res)
            y_test_encoded = le.transform(y_test_res)

            logger.info("Converted ylabels to classes using LabelEncoder")
            logger.info("Data Transformation Finished")
            logger.info("Saving the transformed X_train,X_test,y_train and y_test into Artifact Folder")


            with open(self.data_transformation_config.tfidf_path, 'wb') as file:
                pickle.dump(tfidf, file)

            with open(self.data_transformation_config.lableEncoder_path, 'wb') as file:
                pickle.dump(le, file)

            save_npz(self.data_transformation_config.Xtrain_tfidf_path,X_train_tfidf)
            save_npz(self.data_transformation_config.Xtest_tfidf_path,X_test_tfidf)
            np.save(self.data_transformation_config.ytrain_encoded_path,y_train_encoded)
            np.save(self.data_transformation_config.ytest_encoded_path,y_test_encoded)
            
            logger.info("Saved the transformed X_train,X_test,y_train and y_test into Artifact Folder")

            return (
                X_train_tfidf,
                X_test_tfidf,
                y_train_encoded,
                y_test_encoded,
                self.data_transformation_config.Xtrain_tfidf_path,
                self.data_transformation_config.Xtest_tfidf_path,
                self.data_transformation_config.ytrain_encoded_path,
                self.data_transformation_config.ytest_encoded_path,
                self.data_transformation_config.tfidf_path,
                self.data_transformation_config.lableEncoder_path
            )

        except Exception as e:
            raise CustomException("Something wrong in initiateDataTransformation method of dataTransformation class")