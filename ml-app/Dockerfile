FROM python:3.12-slim
RUN apt-get update && apt-get upgrade -y && apt-get clean 
WORKDIR /app
COPY . /app 
RUN pip install uv 
RUN uv pip install --system -e .
EXPOSE 7860  
RUN python ETL.py
CMD ["python", "app.py"]
