from dataclasses import dataclass 
from typing import ClassVar
from pathlib import Path 

@dataclass 
class DataIngestion:
    ROOT_DIR_PATH: ClassVar[Path]
    DATA_ROOT_DIR_PATH: ClassVar[Path]
    INGESTION_ROOT_DIR_PATH: ClassVar[Path]
    FEATURE_STORE_ROOT_DIR_PATH: ClassVar[Path]
    RAW_DATA_FILE_PATH: ClassVar[Path]
    INGESTED_ROOT_DIR_PATH: ClassVar[Path]
    TRAIN_DATA_FILE_PATH: ClassVar[Path]
    TEST_DATA_FILE_PATH: ClassVar[Path]
    DATABASE_FILE_PATH: ClassVar[Path]
    DATABASE_TABLE_NAME: ClassVar[str]

@dataclass 
class DataValidation:
    ROOT_DIR_PATH:ClassVar[Path]
    DATA_ROOT_DIR_PATH:ClassVar[Path]
    VALIDATION_ROOT_DIR_PATH:ClassVar[Path]
    VALID_ROOT_DIR_PATH:ClassVar[Path]
    VALID_TRAIN_DATA_FILE_PATH:ClassVar[Path]
    VALID_TEST_DATA_FILE_PATH:ClassVar[Path]
    INVALID_ROOT_DIR_PATH:ClassVar[Path]
    INVALID_TRAIN_DATA_FILE_PATH:ClassVar[Path]
    INVALID_TEST_DATA_FILE_PATH:ClassVar[Path]
    REPORT_FILE_FILE_PATH:ClassVar[Path]
    SCHEMA_FILE_PATH:ClassVar[Path]

