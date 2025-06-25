from churn_modelling.components.model_prediction import ModelPredictionComponents
from churn_modelling.configuration import ModelPredictionConfig
from sklearn.compose import ColumnTransformer
from skorch import NeuralNetClassifier
from datetime import datetime
from dataclasses import dataclass 
import pandas as pd 
import numpy as np 


@dataclass 
class PredictionPipeline:
    def run(self, model:NeuralNetClassifier, preprocessor:ColumnTransformer, data:pd.DataFrame, time:datetime) -> np.int64:
        obj = ModelPredictionComponents(ModelPredictionConfig)
        return obj.main(model, preprocessor, data, time)
    
    