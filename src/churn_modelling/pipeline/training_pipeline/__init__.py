from churn_modelling.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from churn_modelling.pipeline.stage_02_data_validation import DataValidationPipeline
from dataclasses import dataclass 


@dataclass
class TrainingPipeline:
    def run(self):
        # data ingestion 
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()

        # data validtaion 
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()

