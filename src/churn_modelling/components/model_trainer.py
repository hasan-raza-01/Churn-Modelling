from dotenv import load_dotenv
load_dotenv()

import torch
from dataclasses import dataclass 
from sklearn.model_selection import GridSearchCV
from churn_modelling.exception import CustomException 
from churn_modelling.logger import logging 
from sklearn.metrics import accuracy_score
from churn_modelling.utils import load_json, create_dirs, dump_json 
from churn_modelling.entity import DataTransformation, ModelTrainer
from churn_modelling.utils.model.functions import get_NeuralNetClassifier 
import numpy as np
import pandas as pd
import sys, mlflow, os


# set mlflow tracking uri
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

@dataclass 
class ModelTrainerComponents:
    data_transformation_config: DataTransformation
    model_trainer_config:ModelTrainer

    def load_data(self):
        try:
            logging.info("In load_data")

            # load transformed data 
            # train data 
            train_data_path = self.data_transformation_config.TRAIN_DATA_FILE_PATH
            train_data = pd.read_csv(train_data_path)
            logging.info(f"train data loaded from {{{train_data_path}}}")
            # test data 
            test_data_path = self.data_transformation_config.TEST_DATA_FILE_PATH
            test_data = pd.read_csv(test_data_path)
            logging.info(f"test data loaded from {{{test_data_path}}}")

            target = self.model_trainer_config.TARGET
            self.X_train, self.y_train, self.X_test, self.y_test = train_data.drop(target, axis=1), train_data[target], test_data.drop(target, axis=1), test_data[target]
            logging.info(f'X_train.shape, y_train.shape, X_test.shape, y_test.shape = {self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape}')

            logging.info("Out load_data")
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)

    def train_and_evaluate(self):
        try:
            logging.info("In train_and_evaluate")

            # Create a NeuralNetClassifier instance that wraps our PyTorch model.
            self.input_dim = self.X_train.shape[1]
            net = get_NeuralNetClassifier(module__input_dim=self.input_dim)

            # load params from directory
            params_path = self.model_trainer_config.PARAMS_FILE_PATH
            params = load_json(params_path)
            logging.info(f"params loaded from {{{params_path}}}")
            params["optimizer"] = [torch.optim.SGD, torch.optim.Adam] 
            logging.info(f"addition on params ---> params[\"optimizer\"] = [torch.optim.SGD, torch.optim.Adam]")

            # mlflow logging 
            with mlflow.start_run():
                # grid search object 
                self.grid = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', n_jobs=-1)

                # fit on grid
                self.grid.fit(self.X_train.astype(np.float32), self.y_train.astype(np.int64))
                logging.info(f"best Score on grid search:{self.grid.best_score_}")

            logging.info("Out train_and_evaluate")
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
        
    def test(self):
        try:
            logging.info("In test")

            # prediction 
            predictions = self.grid.best_estimator_.predict(self.X_test.astype(np.float32))

            # calculate accuracy 
            self.test_score_ = accuracy_score(self.y_test.astype(np.int64), predictions)
            logging.info(f"score on test data {{{self.test_score_}}}")

            logging.info("Out test")
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
        
    def save_outputs(self):
        try:
            logging.info("In save_outputs")

            # save best params
            best_params_path = self.model_trainer_config.BEST_PARAMS_FILE_PATH
            best_params = self.grid.best_params_
            best_params["module__input_dim"] = self.input_dim
            dump_json(best_params, best_params_path)
            logging.info(f"saved best params at {{{best_params_path}}}")

            # save scoreS
            scores_file_path = self.model_trainer_config.SCORES_FILE_PATH
            dump_json({
                "grid_best_score_":self.grid.best_score_,
                "test_score_":self.test_score_
            }, scores_file_path)
            logging.info(f"saved scores at {{{scores_file_path}}}")

            # save model params, optimizer and history
            estimator_weights_path = self.model_trainer_config.ESTIMATOR_WEIGHT_FILE_PATH
            optimizer_path = self.model_trainer_config.OPTIMIZER_FILE_PATH
            estimator_history_path = self.model_trainer_config.ESTIMATOR_HISTORY_FILE_PATH

            self.grid.best_estimator_.save_params(
                f_params=estimator_weights_path,
                f_optimizer=optimizer_path,
                f_history=estimator_history_path
            )
            logging.info(f"saved estimator components at {{{self.model_trainer_config.TRAINING_ROOT_DIR_PATH}}}")
            
            logging.info("Out save_outputs")
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
    
    def main(self):
        # create required directories 
        create_dirs(self.model_trainer_config.ROOT_DIR_PATH)
        create_dirs(self.model_trainer_config.MODEL_ROOT_DIR_PATH)
        create_dirs(self.model_trainer_config.TRAINING_ROOT_DIR_PATH)

        self.load_data()
        mlflow.autolog()
        self.train_and_evaluate()
        self.test()
        self.save_outputs()  

