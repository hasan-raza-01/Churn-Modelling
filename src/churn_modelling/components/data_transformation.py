from churn_modelling.utils import dump_json, load_json, create_dirs, save_pickle
from churn_modelling.entity import DataValidation, DataTransformation
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from churn_modelling.exception import CustomException 
from sklearn.compose import ColumnTransformer
from churn_modelling.logger import logging
from sklearn.pipeline import Pipeline 
from sklearn.utils import resample 
from dataclasses import dataclass 
import pandas as pd
import sys 



@dataclass 
class DataTransformationComponents:
    data_validation_config:DataValidation
    data_transformation_config: DataTransformation 

    def load_data(self):
        try:
            logging.info('In load_data')

            # load validation report
            report_path = self.data_validation_config.REPORT_FILE_FILE_PATH 
            report = load_json(report_path)
            logging.info(f'loaded report from {{{report_path}}}')

            # load the data if validation status of data is true 
            for key, value in report.items():
                if key.strip().lower() == 'train':
                    if value['status']:
                        train_data_path = self.data_validation_config.VALID_TRAIN_DATA_FILE_PATH
                        self.train_data = pd.read_csv(train_data_path)
                        logging.info(f'train data loaded from {{{train_data_path}}}')
                if key.strip().lower() == 'test':
                    if value['status']:
                        test_data_path = self.data_validation_config.VALID_TEST_DATA_FILE_PATH
                        self.test_data = pd.read_csv(test_data_path)
                        logging.info(f'test data loaded from {{{test_data_path}}}') 

            logging.info('Out load_data') 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    def transform_data(self):
        try:
            logging.info('In transform_data')

            # remove duplidates
            self.train_data.drop_duplicates(inplace=True)
            self.test_data.drop_duplicates(inplace=True)
            logging.info(f'droped duplicates')

            # remove unnecessory columns 
            unnecessory_features = ["RowNumber", "CustomerId", "Surname"] 
            self.train_data.drop(unnecessory_features, axis=1, inplace=True)
            self.test_data.drop(unnecessory_features, axis=1, inplace=True)
            logging.info(f'features after drop unnecessory columns\ntrain:{self.train_data.columns}\ntest:{self.test_data.columns}')

            # target column 
            self.target = 'Exited'
            logging.info(f'target column {{{self.target}}}')

            # concatenate both train and test 
            data = pd.concat([self.train_data, self.test_data], axis=0)
            logging.info(f'concatenated both train and test, shape[before:[{self.train_data.shape}, {self.test_data.shape}], after:{data.shape}]')

            # distinguish numerical and categorical features
            numerical_features = [feature for feature in data.columns if data[feature].dtype != "O" and len(data[feature].unique()) > 15 and feature != self.target]
            categorical_features = [feature for feature in data.columns if feature not in numerical_features and feature != self.target]
            logging.info(f'numerical features:{numerical_features}, count:{len(numerical_features)}\ncategorical features:{categorical_features}, count:{len(categorical_features)}')

            # handling of null values
            # numerical features
            # train data 
            for feature in numerical_features:
                self.train_data.loc[:, feature] = self.train_data[feature].fillna(data[feature].mean())
            logging.info('null values handling of numerical features for train data completed')
            # test data 
            for feature in numerical_features:
                self.test_data.loc[:, feature] = self.test_data[feature].fillna(data[feature].mean())
            logging.info('null values handling of numerical features for test data completed')

            # categorical features
            # train data 
            for feature in categorical_features:
                self.train_data.loc[:, feature] = self.train_data[feature].fillna(data[feature].mode()[0])
            logging.info('null values handling of categorical features for train data completed')
            # test data 
            for feature in categorical_features:
                self.test_data.loc[:, feature] = self.test_data[feature].fillna(data[feature].mode()[0])
            logging.info('null values handling of categorical features for test data completed')

            # get categorical feature with type object to transform into numeric 
            categorical_features = [feature for feature in data.columns if data[feature].dtype == 'O' and feature not in numerical_features]
            logging.info(f'categorical values which needs to be transformed from object to numeric {{{categorical_features}}}')

            # pipelines
            numerical_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
            categorical_pipeline = Pipeline([
                ('encoder', OneHotEncoder())
            ])
            logging.info('pipelines created for both numerical and categorical features')

            # final preprocessor 
            self.preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_features),
                ('categorical_pipeline', categorical_pipeline, categorical_features)
            ], remainder='passthrough', n_jobs=-1, verbose=True, verbose_feature_names_out=True)
            logging.info('column transformer initialized')

            # transform data 
            # train data 
            transformed_train_data = self.preprocessor.fit_transform(self.train_data)
            logging.info('successfully transformed train data')
            # test data 
            transformed_test_data = self.preprocessor.fit_transform(self.test_data)
            logging.info('successfully transformed test data')

            # create data frame with transformed data 
            self.columns = [name.split('__')[1] for name in self.preprocessor.get_feature_names_out()]
            logging.info(f'columns after transformation {self.columns}')
            # train data 
            self.transformed_train_data = pd.DataFrame(transformed_train_data, columns=self.columns)
            # test data 
            self.transformed_test_data = pd.DataFrame(transformed_test_data, columns=self.columns)
            logging.info('converted transformed train and test data into dataframes')

            # handle imbalnced training dataset
            # get majority category
            majority_class = {v:k for k, v in self.transformed_train_data[self.target].value_counts().to_dict().items()}[max({v:k for k, v in self.transformed_train_data[self.target].value_counts().to_dict().items()})]
            minurity_class = {v:k for k, v in self.transformed_train_data[self.target].value_counts().to_dict().items()}[min({v:k for k, v in self.transformed_train_data[self.target].value_counts().to_dict().items()})]
            logging.info(f'majority and minurity class of target column {{{majority_class}, {minurity_class}}}')
            # majority_class and minurity_class data 
            train_data_majority_class = self.transformed_train_data[self.transformed_train_data[self.target]==majority_class]
            train_data_minurity_class = self.transformed_train_data[self.transformed_train_data[self.target]==minurity_class]
            logging.info(f'majority class of train data shape:{train_data_majority_class.shape}, minurity class of train data shape:{train_data_minurity_class.shape}')

            # resampled train data of minurity class 
            train_data_minurity_class_resampled = resample(train_data_minurity_class, replace=True, n_samples=len(train_data_majority_class), random_state=42)
            logging.info(f'shape after Upsampling; majority class of train data shape:{train_data_majority_class.shape}, minurity class of train data shape::{train_data_minurity_class_resampled.shape}')

            # concatenate train data of majority and minurity class after resampling  
            self.transformed_train_data_resampled = pd.concat([train_data_majority_class, train_data_minurity_class_resampled], axis=0)

            logging.info('Out transform_data') 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    def save_outputs(self):
        try:
            logging.info('In save_outputs')

            # train data 
            train_data_path = self.data_transformation_config.TRAIN_DATA_FILE_PATH
            self.transformed_train_data_resampled.to_csv(train_data_path, index=False)
            logging.info(f'saved train data at {{{train_data_path}}}')

            # test data 
            test_data_path = self.data_transformation_config.TEST_DATA_FILE_PATH
            self.transformed_test_data.to_csv(test_data_path, index=False)
            logging.info(f'saved test data at {{{test_data_path}}}')

            # feature names
            feature_names_path = self.data_transformation_config.FEATURES_FILE_PATH
            dump_json({'columns':self.columns}, feature_names_path)
            logging.info(f'feature names saved at {{{feature_names_path}}}')

            # preprocessor 
            preprocessor_path = self.data_transformation_config.PREPROCESSOR_FILE_PATH
            save_pickle(preprocessor_path, self.preprocessor)
            logging.info(f'preprocessor saved at {{{preprocessor_path}}}')

            logging.info('Out save_outputs') 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def main(self):
        # create required directoies 
        create_dirs(self.data_transformation_config.ROOT_DIR_PATH) 
        create_dirs(self.data_transformation_config.DATA_ROOT_DIR_PATH) 
        create_dirs(self.data_transformation_config.TRANSFORMATION_ROOT_DIR_PATH)

        self.load_data()
        self.transform_data()
        self.save_outputs()

