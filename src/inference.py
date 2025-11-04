import mlflow
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
import plotly.graph_objects as go
seed_value = 42
import logging
from datetime import timezone
import datetime
from datetime import date
import os 
import io
from google.cloud import aiplatform
import pickle
from google.cloud import storage
from dotenv import load_dotenv
from utils import setup_logger
from read_data import pull_data
#removed src from imports so that docker can corretly import 

#load env vars 
load_dotenv()
API_PATH=os.getenv("API_URL")
LOG_PATH = os.getenv("LOG_PATH")
bucket=os.getenv("MODEL_BUCKET")
model_filename=os.getenv("MODEL_FILENAME")

#setup here because we are not going to call main.py, but dockerise it.
my_logger = setup_logger("my_temperature_logger", LOG_PATH) #log to tmp file in docker cont
#my_logger = logging.getLogger("my_temperature_logger")
my_logger.info(f"{datetime.datetime.now()}:Initialising the inference.py script")

def upload_model_from_gcp(bucket,model_filename):
    """
    Get the model deployed on gcp model registry    
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    blob = bucket.blob(model_filename)
    pickle_bytes = blob.download_as_bytes()
    loaded_model = pickle.load(io.BytesIO(pickle_bytes))
    my_logger.info(f"'Model loaded:', {bucket},{client}")
    return loaded_model

def write_preds_to_gcp_bucket(bucket_name,destination_blob_name,source_file_obj):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    #upload from the file-like object (in memory)
    source_file_obj.seek(0)
    blob.upload_from_file(source_file_obj, content_type='text/csv')
    print(f'File uploaded to {destination_blob_name}.')

#Pull data for a certain hour/day(also pulla actuals)
def pull_data_for_inference(api_path,bucket,model_filename):
    data=pull_data(api_path,'not_all',"2010-01-01") #all:pulls aall historical data, not all pulls current days data 
    #data_path
    #data=data.drop(columns='Unnamed: 0')
    data.date=pd.to_datetime(data.date)
    #filter for current utc hour
    #get current utc hour 
    current_utc_hour=datetime.datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    data=data.query('date==@current_utc_hour')
    #log dims 
    my_logger.info(f"Hour Data Shape:{data.shape}(Shape Should be 24)")
    my_logger.info(f"Hour Data Min:{data.date.min()}")
    my_logger.info(f"Hour Data Max:{data.date.max()}")
    location_map={'Pune':1,'Mississauga':2}
    data.location=data.location.map(location_map)
    #remove temperature col
    y_actual=data.drop(columns=['temperature_2m'])
    data_for_pred=data.drop(columns=['temperature_2m','date'])
    print(data_for_pred)
    #upload model from gcp 
    model=upload_model_from_gcp(bucket,model_filename)
    y_pred=pd.DataFrame(model.predict(data_for_pred))
    y_pred.columns=['preds']
    final_data=pd.concat([y_pred,data_for_pred.reset_index(drop=True)],axis=1)
    print(final_data)
    my_logger.info(f"Y pred Shape:{y_pred.shape}(Shape Should be 2)")
    #log the actuals vs predicted into a gcp bucket 
    #save the filtered df to a CSV in memory
    csv_buffer = io.StringIO()  #source_file_obj
    final_data.to_csv(csv_buffer, index=False)  
    write_preds_to_gcp_bucket(bucket,f'predictions/preds_and_actuals_{current_utc_hour}',csv_buffer)    
    my_logger.info(f"The file has been written to the bucket !")
    my_logger.info(f"inference script is run successfully")

    #mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_pred)), context="Y Pred")
    return y_pred,data


def main():
    y_pred, data = pull_data_for_inference(API_PATH, bucket,model_filename)
    my_logger.info("Inference complete!")

if __name__ == "__main__":
    main()