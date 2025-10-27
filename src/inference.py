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
from dotenv import load_dotenv
from utils import setup_logger
from read_data import pull_data
#removed src from imports so that docker can corretly import 

load_dotenv()
API_PATH=os.getenv("API_URL")
LOG_PATH = os.getenv("LOG_PATH")
models_dir=os.getenv("MODEL_PATH")
#setup here because we are not going to call main.py, but dockerise it.
my_logger = setup_logger("my_temperature_logger", LOG_PATH) #log to tmp file in docker cont
#my_logger = logging.getLogger("my_temperature_logger")
my_logger.info(f"{datetime.datetime.now()}:Initialising the inference.py script")


#Pull data for a certain hour/day(also pulla actuals)
def pull_data_for_inference(api_path,name_of_best_model,models_dir):
    data,data_path=pull_data(api_path,'not_all',"2010-01-01") #all:pulls aall historical data, not all pulls current days data 
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
    #upload model
    model_path = os.path.join(models_dir, f"{name_of_best_model}.pkl")
    with open(f'{model_path}', "rb") as f:
        model = pickle.load(f)
    y_pred=model.predict(data_for_pred)
    my_logger.info(f"Y pred Shape:{y_pred.shape}(Shape Should be 2)")
    my_logger.info(f"inference script is run successfully")
    my_logger.info(f"hello!")

    #mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_pred)), context="Y Pred")
    return y_pred,data


def main():
    y_pred, data = pull_data_for_inference(API_PATH, "XGBoost",models_dir)
    my_logger.info("Inference complete!")

if __name__ == "__main__":
    main()