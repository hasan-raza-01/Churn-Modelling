from churn_modelling.constants import (
    DataIngestionConstants,
    DataValidationConstants, 
    DataTransformationConstants
) 
from dataclasses import dataclass
from pathlib import Path
import os  

@dataclass 
class DataIngestionConfig:
    ROOT_DIR_PATH = Path(DataIngestionConstants.ROOT_DIR)
    DATA_ROOT_DIR_PATH = Path(os.path.join(ROOT_DIR_PATH, DataIngestionConstants.DATA_ROOT_DIR))
    INGESTION_ROOT_DIR_PATH = Path(os.path.join(DATA_ROOT_DIR_PATH, DataIngestionConstants.INGESTION_ROOT_DIR))
    FEATURE_STORE_ROOT_DIR_PATH = Path(os.path.join(INGESTION_ROOT_DIR_PATH, DataIngestionConstants.FEATURE_STORE_ROOT_DIR))
    RAW_DATA_FILE_PATH = Path(os.path.join(FEATURE_STORE_ROOT_DIR_PATH, DataIngestionConstants.RAW_DATA_FILE_NAME))
    INGESTED_ROOT_DIR_PATH = Path(os.path.join(INGESTION_ROOT_DIR_PATH, DataIngestionConstants.INGESTED_ROOT_DIR))
    TRAIN_DATA_FILE_PATH = Path(os.path.join(INGESTED_ROOT_DIR_PATH, DataIngestionConstants.TRAIN_DATA_FILE_NAME))
    TEST_DATA_FILE_PATH = Path(os.path.join(INGESTED_ROOT_DIR_PATH, DataIngestionConstants.TEST_DATA_FILE_NAME))
    DATABASE_FILE_PATH = Path(DataIngestionConstants.DATABASE_FILE_PATH)
    DATABASE_TABLE_NAME = DataIngestionConstants.DATABASE_TABLE_NAME

@dataclass 
class DataValidationConfig:
    ROOT_DIR_PATH = Path(DataValidationConstants.ROOT_DIR)
    DATA_ROOT_DIR_PATH = Path(os.path.join(ROOT_DIR_PATH, DataValidationConstants.DATA_ROOT_DIR))
    VALIDATION_ROOT_DIR_PATH = Path(os.path.join(DATA_ROOT_DIR_PATH, DataValidationConstants.VALIDATION_ROOT_DIR))
    VALID_ROOT_DIR_PATH = Path(os.path.join(VALIDATION_ROOT_DIR_PATH, DataValidationConstants.VALID_ROOT_DIR))
    VALID_TRAIN_DATA_FILE_PATH = Path(os.path.join(VALID_ROOT_DIR_PATH, DataValidationConstants.TRAIN_DATA))
    VALID_TEST_DATA_FILE_PATH = Path(os.path.join(VALID_ROOT_DIR_PATH, DataValidationConstants.TEST_DATA))
    INVALID_ROOT_DIR_PATH = Path(os.path.join(VALIDATION_ROOT_DIR_PATH, DataValidationConstants.INVALID_ROOT_DIR))
    INVALID_TRAIN_DATA_FILE_PATH = Path(os.path.join(INVALID_ROOT_DIR_PATH, DataValidationConstants.TRAIN_DATA))
    INVALID_TEST_DATA_FILE_PATH = Path(os.path.join(INVALID_ROOT_DIR_PATH, DataValidationConstants.TEST_DATA))
    REPORT_FILE_FILE_PATH = Path(os.path.join(VALIDATION_ROOT_DIR_PATH, DataValidationConstants.REPORT_FILE))
    SCHEMA_FILE_PATH = Path(DataValidationConstants.SCHEMA_FILE_PATH)

@dataclass 
class DataTransformationConfig:
    ROOT_DIR_PATH = Path(DataTransformationConstants.ROOT_DIR)
    DATA_ROOT_DIR_PATH = Path(os.path.join(ROOT_DIR_PATH, DataTransformationConstants.DATA_ROOT_DIR))
    TRANSFORMATION_ROOT_DIR_PATH = Path(os.path.join(DATA_ROOT_DIR_PATH, DataTransformationConstants.TRANSFORMATION_ROOT_DIR))
    TRAIN_DATA_FILE_PATH = Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TRAIN_DATA))
    TEST_DATA_FILE_PATH = Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.TEST_DATA))
    FEATURES_FILE_PATH = Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.FEATURES))
    PREPROCESSOR_FILE_PATH = Path(os.path.join(TRANSFORMATION_ROOT_DIR_PATH, DataTransformationConstants.PREPROCESSOR))

