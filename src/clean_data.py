import mlflow
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import random
from src.utils import setup_logger
import datetime
from datetime import date
import logging
from sklearn.preprocessing import MinMaxScaler

my_logger = logging.getLogger("my_temperature_logger")

def data_clean(path):
    """
    Define a function to read in the data, clean, and split into train and test,
    and also scale the data for linear models
    """
    data=pd.read_csv(path).drop(columns='Unnamed: 0')
    data.date=pd.to_datetime(data.date)
    location_map={'Pune':1,'Mississauga':2}
    data.location=data.location.map(location_map)
    return data


def split_data(data,train_upper_date,test_lower_date):
    train_data=data.query(f'date<="{train_upper_date}"') #train_upper_date
    test_data=data.query(f'date>="{test_lower_date}"') #test_lower_date
    train_data.set_index('date',inplace=True)
    test_data.set_index('date',inplace=True)
    my_logger.info(f"{datetime.datetime.now()}:Testmin Min max date{test_data.index.min(),test_data.index.max()}")
    my_logger.info(f"{datetime.datetime.now()}:Train Min  max date{train_data.index.min(),train_data.index.max()}")
    my_logger.info(f"{datetime.datetime.now()}:Test Shape {test_data.shape}")
    my_logger.info(f"{datetime.datetime.now()}:Train Shape {train_data.shape}")
    X_train=train_data.drop(columns='temperature_2m')
    y_train=train_data['temperature_2m']
    X_test=test_data.drop(columns='temperature_2m')
    y_test=test_data['temperature_2m']
    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    my_logger.info(f"X_train Shape:{X_train.shape}")
    my_logger.info(f"X_test Shape:{X_test.shape}")
    my_logger.info(f"X_train_scaled Shape:{X_train_scaled.shape}")
    my_logger.info(f"X_test_scaled Shape:{X_test_scaled.shape}")
    my_logger.info(f"y_train Shape:{y_train.shape}")
    my_logger.info(f"y_test Shape:{y_test.shape}")
    my_logger.info(f"Type X train Shape:{type(X_train)}")
    #log data to mlflow-test, train, data
    mlflow.log_input(mlflow.data.from_pandas(X_train), context="training_features")
    mlflow.log_input(mlflow.data.from_pandas(X_test), context="testing_features")
    mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_train)), context="Y train")
    mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_test)), context="Y test")
    my_logger.info(f"{datetime.datetime.now()}:Data Cleaned and Split")
    return X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test