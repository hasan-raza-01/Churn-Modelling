from churn_modelling.configuration import DataTransformationConfig, ModelTrainerConfig
from churn_modelling.components.model_trainer import ModelTrainerComponents
from dataclasses import dataclass 


@dataclass 
class ModelTrainerPipeline:
    def run(self):
        obj = ModelTrainerComponents(DataTransformationConfig, ModelTrainerConfig)
        obj.main()



if __name__ == '__main__':
    data_transformation_pipeline = ModelTrainerPipeline()
    data_transformation_pipeline.run()

