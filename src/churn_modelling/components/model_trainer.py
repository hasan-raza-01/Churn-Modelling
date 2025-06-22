from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass 
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from churn_modelling.exception import CustomException 
from churn_modelling.logger import logging 
from sklearn.metrics import accuracy_score
from churn_modelling.utils import load_json, create_dirs, dump_json, save_pickle
from churn_modelling.entity import DataTransformation, ModelTrainer
import numpy as np
import pandas as pd
import sys, mlflow, os


# set mlflow tracking uri
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Define a custom neural network that supports a variable number of hidden layers.
class ClassifierModule(nn.Module):
    """
    A flexible neural network for a classification task that supports a tunable number
    of hidden layers. The network architecture is defined as:
    
    Input -> [Hidden Layer 1 -> ReLU -> Dropout] -> ... -> [Hidden Layer N -> ReLU -> Dropout] -> Output Layer
    
    Attributes:
      input_dim (int): Number of input features.
      num_hidden_layers (int): Number of hidden layers in the network.
      hidden_units (int): Number of neurons in each hidden layer.
      dropout (float): Dropout probability applied after each hidden layer activation.
    """
    def __init__(self, input_dim=20, num_hidden_layers=1, hidden_units=50, dropout=0.5):
        # Initialize the parent nn.Module class.
        super().__init__()
        
        # Create a list to hold our hidden layers. We'll use the ModuleList container so that the layers
        # are registered as submodules (required for proper parameter tracking during training).
        hidden_layers = []
        
        # Add the first hidden layer: From input_dim to hidden_units.
        hidden_layers.append(nn.Linear(input_dim, hidden_units))
        
        # Add additional hidden layers (if any) where each receives hidden_units as input and outputs hidden_units.
        # We subtract one because the first layer is already added.
        for _ in range(num_hidden_layers - 1):
            hidden_layers.append(nn.Linear(hidden_units, hidden_units))
        
        # Save the list of hidden layers in a ModuleList so that it is properly managed.
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        # Define a Dropout layer applied after each hidden layer activation.
        self.dropout = nn.Dropout(dropout)
        
        # The final output layer maps the last hidden layer's output to the number of classes.
        # Here, 2 is used for binary classification.
        self.output_layer = nn.Linear(hidden_units, 2)
    
    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the network.
        
        Arguments:
          x (Tensor): Input tensor of shape (batch_size, input_dim)
          
        Returns:
          Tensor: Logits output from the network.
        """
        # If keyword arguments are provided, assume they are features and extract their values.
        # This will convert the keys (e.g., 'CreditScore', …) into a tensor.
        if kwargs:
            # Assuming all columns are numeric and should be concatenated along the feature axis.
            x = torch.tensor([list(sample) for sample in zip(*kwargs.values())])
        else:
            x = args[0]  # usual case if input is a tensor

        # Pass the input through each hidden layer block.
        for layer in self.hidden_layers:
            x = layer(x)       # Apply the linear transformation.
            x = F.relu(x)      # Pass through ReLU activation to introduce non-linearity.
            x = self.dropout(x)  # Apply dropout to reduce overfitting.
        
        # Pass the result from the last hidden layer block into the output layer.
        x = self.output_layer(x)
        # Note: We do not use softmax here; nn.CrossEntropyLoss expects raw logits.
        return x

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
            net = NeuralNetClassifier(
            module=ClassifierModule, 
            criterion=nn.CrossEntropyLoss, 
            optimizer=torch.optim.Adam, 
            optimizer__weight_decay=0.01, 
            max_epochs=10, 
            lr=0.01, 
            batch_size=64, 
            module__input_dim=self.X_train.shape[1], 
            module__num_hidden_layers=1, 
            module__hidden_units=50, 
            module__dropout=0.5, 
            iterator_train__shuffle=True, 
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            verbose=1 
        )

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
            dump_json(self.grid.best_params_, best_params_path)
            logging.info(f"saved best params at {{{best_params_path}}}")

            # save scoreS
            scores_file_path = self.model_trainer_config.SCORES_FILE_PATH
            dump_json({
                "grid_best_score_":self.grid.best_score_,
                "test_score_":self.test_score_
            }, scores_file_path)
            logging.info(f"saved scores at {{{scores_file_path}}}")

            # save model
            estimator_path = self.model_trainer_config.ESTIMATOR_FILE_PATH
            save_pickle(estimator_path, self.grid.best_estimator_)
            logging.info(f"saved estimator at {{{estimator_path}}}")

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

