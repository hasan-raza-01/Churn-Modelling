from churn_modelling.configuration import ModelTrainerConfig, DataTransformationConfig
from churn_modelling.pipeline.training_pipeline import TrainingPipeline
from churn_modelling.pipeline.prediction_pipeline import PredictionPipeline
from churn_modelling.utils.model.functions import get_NeuralNetClassifier
from churn_modelling.utils import load_pickle, load_json 
from churn_modelling.logger import logging 
from dataclasses import dataclass 
from datetime import datetime
import pandas as pd 
import numpy as np  
import gradio as gr
import os 


@dataclass 
class App:
    training_pipeline = TrainingPipeline()
    model_trainer_config = ModelTrainerConfig()
    data_transformation_config = DataTransformationConfig() 

    # training function
    def train_model(self):
        try:
            # run training pipeline 
            self.training_pipeline.run()

            # load model 
            self.load_model()
            
            # load preprocessor
            self.load_preprocessor() 

            return "✅ Model trained successfully!"
        except Exception as e:
            return f"❌ Training failed: {str(e)}"
        
    # load model 
    def load_model(self):
        # load model components and initialize 
        self.model_params = load_json(self.model_params_path)
        logging.info(f"loaded params from {{{self.model_params_path}}}")
        self.weights_path = self.model_trainer_config.ESTIMATOR_WEIGHT_FILE_PATH
        self.optimizer_path = self.model_trainer_config.OPTIMIZER_FILE_PATH
        self.model_history_path = self.model_trainer_config.ESTIMATOR_HISTORY_FILE_PATH
        self.model = get_NeuralNetClassifier(**self.model_params)
        self.model.initialize()
        self.model.load_params(
            f_params=self.weights_path,
            f_optimizer=self.optimizer_path,
            f_history=self.model_history_path
        )
        logging.info(f"loaded and initialize model successfully") 

    # load preprocessor
    def load_preprocessor(self):
        # load preprocessor 
        self.preprocessor_path = self.data_transformation_config.PREPROCESSOR_FILE_PATH
        self.preprocessor = load_pickle(self.preprocessor_path)
        logging.info(f"loaded preprocessor from {{{self.preprocessor_path}}}")

    # initialize required variables 
    def initialize_variables(self):
        # load params  
        self.model_params_path = self.model_trainer_config.BEST_PARAMS_FILE_PATH 

        # if path not available 
        if not os.path.exists(self.model_params_path):
            os.system("dvc pull")
            os.system("dvc repro")
            os.system("dvc push")

        # load model 
        self.load_model()
        
        # load preprocessor
        self.load_preprocessor() 

        # load features info 
        self.features_info = load_json(self.data_transformation_config.FEATURES_FILE_PATH)
        self.input_columns = list(self.features_info.keys())[:-1]

    # prediction function 
    def predict_churn(self, *inputs) -> np.int64:
        inputs = pd.DataFrame([list(inputs)], columns=self.input_columns)
        pred_pipeline = PredictionPipeline()
        prediction =  pred_pipeline.run(self.model, self.preprocessor, inputs, datetime.now().strftime("%H:%M:%S"))
        return "Churn" if prediction == 1 else "Not Churn"
    

if __name__ == "__main__":
    app_obj = App()
    app_obj.initialize_variables()

    # create proper input method from user through gradio 
    inputs = []
    for col in app_obj.input_columns:
        if app_obj.features_info[col][0] > 3:
            inputs.append(gr.Number(label=col))
        else:
            inputs.append(gr.Dropdown(choices=app_obj.features_info[col][1], label=col))

    # gradio interface 
    with gr.Blocks() as app:
        gr.Markdown("## 🧠 Customer Churn Predictor")

        with gr.Row():
            with gr.Column():
                for inp in inputs:
                    inp.render()
                predict_btn = gr.Button("Predict Churn")
            prediction = gr.Textbox(label="Prediction")

        predict_btn.click(fn=app_obj.predict_churn, inputs=inputs, outputs=prediction)

        with gr.Accordion("⚙️ Train the Model", open=False):
            train_btn = gr.Button("Run Training Pipeline")
            train_output = gr.Textbox(label="Training Status", interactive=False)
            train_btn.click(fn=app_obj.train_model, outputs=train_output)

    app.launch(server_name="0.0.0.0", server_port=7860)


