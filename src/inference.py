import mlflow
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
import plotly.graph_objects as go
seed_value = 42
from src.utils import setup_logger
import datetime
from datetime import date
my_logger = setup_logger('my_temperature_logger', 'logs_for_ml.log')
my_logger.info(f"{datetime.datetime.now()}:Initialising the inference.py script")
from src.read_data import pull_data
import logging
from datetime import datetime, timezone
import os 


#Pull data for a certain hour/day(also pulla actuals)
def pull_data_for_inference(api_path,name_of_best_model):
    data_path=pull_data(api_path,'not_all',"2010-01-01") #all:pulls aall historical data, not all pulls current days data 
    data=pd.read_csv(data_path).drop(columns='Unnamed: 0')
    data.date=pd.to_datetime(data.date)
    #filter for current utc hour
    #get current utc hour 
    current_utc_hour=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
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
    models_dir = "./models"
    model_path = os.path.join(models_dir, f"{name_of_best_model}.pkl")
    with open(f'{model_path}', "rb") as f:
        model = pickle.load(f)
    y_pred=model.predict(data_for_pred)
    my_logger.info(f"Y pred Shape:{y_pred.shape}(Shape Should be 2)")
    mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_pred)), context="Y Pred")


 
# #%%
# import pandas as pd
# from datetime import datetime, timezone
# from read_data import pull_data
# data_path=pull_data("https://archive-api.open-meteo.com/v1/archive",'not_all',"2010-01-01") #all:pulls aall historical data, not all pulls current days data 
# data=pd.read_csv('./data/temp_data_not_all_2025-10-23.csv').drop(columns='Unnamed: 0')
# data.date=pd.to_datetime(data.date)
# #filter for current utc hour
# #get current utc hour 
# current_utc_hour=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
# print(current_utc_hour)
# data=data.query('date== @current_utc_hour')
# data.head()




#Clean data(locationis all)
#Upload Model
#Write preds and actuals 
#Log the above data in a db-keep appending to that db 
#Plot the actual vs predicted by location in a dashboard 

# %%
