# Churn Modelling

## Workflows

1. Update config.yaml
2. Update .env[optional]
3. Update the constants
4. Update the entity
5. Update the configuration
6. Update the components
7. Update the pipeline
9. Update the dvc.yaml


## launch mlflow tracking server 'bash command'
```
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://<your-bucket-name>/<directory>/ \
  --host 0.0.0.0 \
  --port 5000
```

## changes needed before deployment 
- mlflow tracking uri 
- --default-artifact-root
