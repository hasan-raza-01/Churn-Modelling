from churn_modelling.components.data_transformation import DataTransformationComponents
from churn_modelling.configuration import DataValidationConfig, DataTransformationConfig
from dataclasses import dataclass 


@dataclass 
class DataTransformationPipeline:
    def run(self):
        obj = DataTransformationComponents(DataValidationConfig, DataTransformationConfig)
        obj.main()



if __name__ == '__main__':
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.run()

