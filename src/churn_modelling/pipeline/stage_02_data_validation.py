from churn_modelling.configuration import (
    DataIngestionConfig,
    DataValidationConfig
)
from churn_modelling.components.data_validation import DataValidationComponents
from dataclasses import dataclass 


@dataclass 
class DataValidationPipeline:
    def run(self):
        obj = DataValidationComponents(DataIngestionConfig, DataValidationConfig)
        obj.main()


if __name__ == '__main__':
    data_validation_pipeline = DataValidationPipeline()
    data_validation_pipeline.run()

