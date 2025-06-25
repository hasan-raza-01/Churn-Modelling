FROM python:3.12-slim
RUN apt-get update && apt-get upgrade -y && apt-get clean 
WORKDIR /app
RUN pip install uv 
RUN uv pip install --system mlflow boto3 
EXPOSE 5000
CMD mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root s3://projectsbucket01/ChurnModelling-mlruns/ \
  --host 0.0.0.0 \
  --port 5000