# Churn Modelling
This project showcases a full-cycle machine learning system for customer churn prediction, emphasizing modularity, traceability, and production-readiness.

At its core is a PyTorch-based artificial neural network (ANN), wrapped with Skorch to integrate cleanly with scikit-learn and enable hyperparameter tuning via GridSearchCV. The application includes an interactive Gradio interface for real-time predictions, and leverages an MLOps stack to manage experiment tracking, versioning, and deployment.

🔧 Key Features
- Modeling: Torch + Skorch ANN trained and optimized using GridSearchCV.
- Experiment Tracking: Integrated with MLflow, logging runs and artifacts to a remote S3 bucket.
- Versioning with DVC: Complete data and artifact tracking using DVC, backed by remote S3 storage.
- Prediction Management: Inference outputs are persisted to a dedicated S3 object for traceability.
- User Interface: A smooth Gradio app allows hands-on interaction with the pipeline.
- Deployment: Entire setup is containerized using Docker and orchestrated with Docker Compose for consistent, reproducible execution.

## environment variables
```
S3_BUCKET=""
S3_BUCKET_OBJECT=""
S3_BUCKET_PREDICTION_OBJECT=""
S3_BUCKET_MLFLOW_DIR="ChurnModelling-mlruns"
MLFLOW_S3_ENDPOINT_URL="https://s3.amazonaws.com"
MLFLOW_TRACKING_URI="http://mlflow-server:5000"
```

## launch mlflow tracking server [before running app.py "bash command"]
```
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root s3://<your-bucket-name>/<directory/path>/ \
  --host 0.0.0.0 \
  --port 5000
```

## MLFLOW_TRACKING_URI on .env
### local machine 
```
http://localhost:5000
```
### dockerize/deploy 
```
http://mlflow-server:5000
```

## env file path [env_file] in docker-compose.yml to 
### local machine
```
.env
```
### dockerize/deploy 
```
/home/ubuntu/.env
```
