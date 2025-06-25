from churn_modelling.exception import CustomException
from churn_modelling.logger import logging
from botocore.exceptions import ClientError
from dataclasses import dataclass, field
from typing import Optional, Any
import boto3, os, sys



@dataclass
class S3_Cloud:
    """AWS S3 object

    Args:
        file_path (str): File to upload
        bucket (str): Bucket to upload to
    """
    bucket: str
    object_name: Optional[str] = None
    s3_client: Any = field(init=False)  # Added type annotation

    def __post_init__(self) -> None:
        """Initializes S3 client"""
        self.s3_client = boto3.client('s3')

    def upload_file(self, file_path:str) -> bool:
        """Upload a file to an S3 bucket

        Args:
            object_name[str]=None: S3 object name. If not specified then file_path is used

        Returns:
            True if file was uploaded, else False
        """
        logging.info("In upload_file")

        # If S3 object_name was not specified, use file_path
        if self.object_name is None:
            self.object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, self.bucket, self.object_name)
            logging.info("data uploaded successfully")
        except ClientError as e:
            logging.error(e)
            return False
        
        logging.info("Out upload_file")
        return True


    def download_file(self, file_path:str) -> bool:
        """downloads file to the given file path with file extention

        Args:
            file_path (str): file to read from cloud

        Returns:
            True if download successful, else False
        """
        logging.info("In download_file")

        try:
            self.s3_client.download_file(self.bucket, self.object_name, file_path)
        except ClientError as e:
            logging.error(e)
            return False
        
        logging.info("Out download_file")
        return True

