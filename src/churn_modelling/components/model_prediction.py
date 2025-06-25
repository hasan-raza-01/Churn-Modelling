from churn_modelling.utils import dump_json, load_json, create_dirs
from churn_modelling.exception import CustomException 
from churn_modelling.entity import ModelPrediction
from sklearn.compose import ColumnTransformer
from churn_modelling.logger import logging 
from churn_modelling.cloud import S3_Cloud
from skorch import NeuralNetClassifier
from dataclasses import dataclass 
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path 
import numpy as np
import pandas as pd 
import sys, os 


@dataclass 
class ModelPredictionComponents:
    model_prediction_config:ModelPrediction 

    def predict(self):
        try:
            logging.info("In predict")

            # transform data and perform prediction
            columns = [name.split('__')[1] for name in self.preprocessor.get_feature_names_out()]
            transformed_data = self.preprocessor.transform(self.data)
            model_input = pd.DataFrame(transformed_data, columns=columns).astype(np.float32)
            self.prediction = self.model.predict(model_input)[0]

            logging.info("Out predict")
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys) 
        
    def save_outputs(self, time:datetime):
        try:
            logging.info("In save_outputs")
            time_stamp = datetime.now().strftime("%d_%m_%Y")
            dir_path, file_name = os.path.split(self.model_prediction_config.FILE_PATH)
            self.output_file_path = Path(os.path.join(dir_path, time_stamp + file_name))
            
            # load prevous predictions from cloud if not available in local 
            if not os.path.exists(self.output_file_path):
                self.pull_from_cloud()
            
            output = {
                time:{
                    "input":self.data.values.tolist(),
                    "output":self.prediction
                }
            }

            # load prevoiusly saved data if available 
            if os.path.exists(self.output_file_path):
                output.update(load_json(self.output_file_path))

            # save data into file 
            dump_json(output, self.output_file_path)
            logging.info(f"saved outputs at {{{self.output_file_path}}}")

            logging.info("Out save_outputs")
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys) 
        
    def push_to_cloud(self):
        try:
            logging.info("In push_to_cloud") 
            load_dotenv()

            cloud=S3_Cloud(
                bucket=os.getenv("S3_BUCKET"),
                object_name=os.getenv("S3_BUCKET_PREDICTION_OBJECT")
            )
            status = cloud.upload_file(self.output_file_path)
            logging.info(f"push status {{{status}}}")
            logging.info("Out push_to_cloud")
        except Exception as e:
            raise CustomException(e, sys)
        
    def pull_from_cloud(self) -> bool:
        try:
            logging.info("In pull_from_cloud") 
            load_dotenv()
            
            cloud=S3_Cloud(
                bucket=self.model_prediction_config.BUCKET,
                object_name=self.model_prediction_config.OBJECT
            )
            status = cloud.download_file(self.output_file_path)
            logging.info(f"push status {{{status}}}")
            logging.info("Out push_to_cloud")
        except:
            pass 
        
    def main(self, model:NeuralNetClassifier, preprocessor:ColumnTransformer, data:pd.DataFrame, time:datetime) -> np.int64:
        # create required directories
        create_dirs(self.model_prediction_config.ROOT_DIR_PATH)

        self.model = model
        self.preprocessor = preprocessor
        self.data = data
        self.predict()
        self.save_outputs(time)
        self.push_to_cloud()

        return self.prediction 
    
