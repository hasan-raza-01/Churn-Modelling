from churn_modelling.entity import DataIngestion 
from churn_modelling.exception import CustomException 
from sklearn.model_selection import train_test_split 
from churn_modelling.utils import create_dirs 
from botocore.exceptions import ClientError
from churn_modelling.logger import logging 
from dataclasses import dataclass 
import sys , sqlite3, boto3, os 
from dotenv import load_dotenv 
import pandas as pd 



@dataclass 
class DataIngestionComponents:
    data_ingestion_config:DataIngestion 
    
    def data_collection(self):
        try:
            logging.info("In data_collection") 
            # load .env 
            load_status = load_dotenv(".env")
            logging.info(f".env load status {{{load_status}}}")
            # download db file from s3
            try:
                self.database_path = self.data_ingestion_config.DATABASE_FILE_PATH
                # create directories for data if not avalilable
                directories, _ = os.path.split(self.database_path)
                os.makedirs(directories, exist_ok=True)
                # create s3 client 
                self.s3_client = boto3.client('s3')
                # download file 
                logging.info("downloading db file..........")
                self.s3_client.download_file(os.getenv('S3_BUCKET'), os.getenv('S3_BUCKET_OBJECT'), self.database_path)
                logging.info(f"download completed, saved file at {self.database_path}")
            except ClientError as e:
                raise e
            logging.info("Out data_collection") 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def data_conversion(self):
        try:
            logging.info("In data_conversion") 
            
            self.table_name = self.data_ingestion_config.DATABASE_TABLE_NAME

            # create connection with database
            logging.info(f'connecting with database at {{{self.database_path}}}') 
            connection = sqlite3.connect(self.database_path)
            cursor = connection.cursor()
            logging.info(f'connection successful.')

            # fetch all column names from the table
            cursor.execute("SELECT * FROM ChurnModelling LIMIT 1")
            self.columns = [description[0] for description in cursor.description]
            logging.info(f'fetched column names from database {{{self.columns}}}')

            # fetch all data from the table
            cursor.execute(f'SELECT * FROM {self.table_name}')
            data = cursor.fetchall()
            logging.info(f'fetched {len(data)} records from database.')

            # convert sql data into dataframe
            self.data = pd.DataFrame(data, columns=self.columns)
            logging.info('successfully converted database fetched data into dataframe.')

            # commit and close connection 
            connection.commit()
            connection.close()
            logging.info('database connection commited and closed.')

            logging.info("Out data_conversion") 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
            
    def data_splitting(self):
        try:
            logging.info("In data_splitting") 
            
            # split the data into train and test 
            train_data_array, test_data_array = train_test_split(self.data, test_size=0.33, random_state=42)
            self.train_data = pd.DataFrame(train_data_array, columns=self.columns)
            self.test_data = pd.DataFrame(test_data_array, columns=self.columns)

            logging.info("Out data_splitting") 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def save_data(self):
        try:
            logging.info("In save_data") 
            
            # save raw data 
            self.raw_data_path = self.data_ingestion_config.RAW_DATA_FILE_PATH 
            self.data.to_csv(self.raw_data_path, index=False)
            logging.info(f'raw data saved at \'{self.raw_data_path}\'')
            
            # save train data 
            self.train_data_path = self.data_ingestion_config.TRAIN_DATA_FILE_PATH 
            self.train_data.to_csv(self.train_data_path, index=False)
            logging.info(f'train data saved at \'{self.train_data_path}\'')
            
            # save test data 
            self.test_data_path = self.data_ingestion_config.TEST_DATA_FILE_PATH 
            self.test_data.to_csv(self.test_data_path, index=False)
            logging.info(f'test data saved at {{{self.test_data_path}}}')

            logging.info("Out save_data") 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def main(self):
        # create directories 
        create_dirs(self.data_ingestion_config.ROOT_DIR_PATH)
        create_dirs(self.data_ingestion_config.DATA_ROOT_DIR_PATH)
        create_dirs(self.data_ingestion_config.INGESTION_ROOT_DIR_PATH)
        create_dirs(self.data_ingestion_config.FEATURE_STORE_ROOT_DIR_PATH)
        create_dirs(self.data_ingestion_config.INGESTED_ROOT_DIR_PATH)

        self.data_collection()
        self.data_conversion()
        self.data_splitting()
        self.save_data()

