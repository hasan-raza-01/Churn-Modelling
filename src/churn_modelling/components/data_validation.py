from churn_modelling.utils import load_yaml, dump_json, create_dirs
from churn_modelling.entity import DataIngestion, DataValidation
from churn_modelling.exception import CustomException 
from churn_modelling.logger import logging 
from dataclasses import dataclass 
import pandas as pd 
import sys 



@dataclass 
class DataValidationComponents:
    data_ingestion_config:DataIngestion
    data_validation_config:DataValidation 

    def load_data(self):
        try:
            logging.info('In load_data')

            # read train data from artifacts 
            self.train_data_path = self.data_ingestion_config.TRAIN_DATA_FILE_PATH
            self.train_data = pd.read_csv(self.train_data_path)
            logging.info(f'loaded train data from {{{self.train_data_path}}}')

            # read test data from artifacts 
            self.test_data_path = self.data_ingestion_config.TEST_DATA_FILE_PATH
            self.test_data = pd.read_csv(self.test_data_path)
            logging.info(f'loaded test data from {{{self.test_data_path}}}')

            logging.info('Out load_data')
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    def validate_data(self, data:pd.DataFrame) -> dict[str, bool | dict]:
        """creates validation report for given data

        Args:
            data (pd.DataFrame): dataframe object of data which needs to be validated

        Returns:
            dict: keys[status, loaded_schema, generated_schema]
            - status: True if loaded_schema(pre-defined schema) == generated_schema else False 
            - loaded_schema: pre-defined schema which was loaded to compare 
            - generated_schema: schema generated from data which was provided 
        """
        try:
            logging.info('In validate_data')

            # load schema 
            loaded_schema = load_yaml(self.data_validation_config.SCHEMA_FILE_PATH)

            # generate fresh schema of data 
            schema = dict()
            columns_with_dtype = dict()
            numerical_columns = list()

            for col in data.columns:
                columns_with_dtype[col] = str(data[col].dtype)
                if data[col].dtype!="O":
                    numerical_columns.append(col)

            schema["columns"] = columns_with_dtype
            schema["numerical_columns"] = numerical_columns

            status = schema == loaded_schema

            logging.info('Out validate_data')
            return {
                'status':status, 
                'loaded_schema':dict(loaded_schema),
                'generated_schema':schema 
            }
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def generate_report(self, train_validation_info:dict, test_validation_info:dict):
        """generates final validation report for train and test data

        Args:
            train_validation_info (dict): output of validate_data when train data is provided to the function
            test_validation_info (dict): output of validate_data when test data is provided to the function
        """
        try:
            logging.info('In generate_report')

            # generate report
            self.validation_report = {
                'train':train_validation_info,
                'test':test_validation_info
            }

            logging.info('Out generate_report')
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def save_outputs(self):
        try:
            logging.info('In save_outputs')

            paths = {
                'valid':{
                    'train':self.data_validation_config.VALID_TRAIN_DATA_FILE_PATH, 
                    'test':self.data_validation_config.VALID_TEST_DATA_FILE_PATH
                },
                'invalid':{
                    'train':self.data_validation_config.INVALID_TRAIN_DATA_FILE_PATH, 
                    'test':self.data_validation_config.INVALID_TEST_DATA_FILE_PATH
                }
            }
            data = {
                'train':self.train_data,
                'test':self.test_data
            }
            for key in self.validation_report.keys():
                go = 'invalid'
                if self.validation_report[key]['status']:
                    go = 'valid'

                # save data to its validation path 
                data[key].to_csv(paths[go][key], index=False)
                logging.info(f'saved {key} data at {{{paths[go][key]}}}')

            # save validation report 
            validation_report_path = self.data_validation_config.REPORT_FILE_FILE_PATH
            dump_json(self.validation_report, validation_report_path)
            logging.info(f'saved validation report to {{{validation_report_path}}}')

            logging.info('Out save_outputs')
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def main(self):
        # create required directories 
        create_dirs(self.data_validation_config.ROOT_DIR_PATH)
        create_dirs(self.data_validation_config.DATA_ROOT_DIR_PATH)
        create_dirs(self.data_validation_config.VALIDATION_ROOT_DIR_PATH)
        create_dirs(self.data_validation_config.VALID_ROOT_DIR_PATH)
        create_dirs(self.data_validation_config.INVALID_ROOT_DIR_PATH)

        self.load_data()
        train_validation_info = self.validate_data(self.train_data)
        test_validation_info = self.validate_data(self.test_data)
        self.generate_report(train_validation_info, test_validation_info)
        self.save_outputs() 

