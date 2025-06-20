from churn_modelling.configuration import DataIngestionConfig
from churn_modelling.components.data_ingestion import DataIngestionComponents 
from dataclasses import dataclass 


@dataclass 
class DataIngestionPipeline:
    def run(self):
        obj = DataIngestionComponents(DataIngestionConfig)
        obj.main()


if __name__ == '__main__':
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()

