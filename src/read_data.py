"""
This file will read in data from the hitorical api. 
We will use this data to train the model
#think about improvements-make things dynamic/read from congif
#make the below a function
#log the data file in mlflow
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from src.utils import setup_logger
import logging
from datetime import datetime, timezone
import os 
# from dotenv import load_dotenv
# load_dotenv()
#https://open-meteo.com/en/features#available_apis

my_logger = logging.getLogger("my_temperature_logger")
my_logger.info(f"{datetime.now()}:Initialising the API connection")
def pull_data(data_url,all_data_or_current_hour,start_date):
    """
    Pulls in data of the full time range, or for the current hour
    Give arg "all" if you want data to be pulled from 2010 to current hour, else hour
    """
    try:     
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        #if we pull all the data, set start and end date, else set start and end date to current hour 
        if all_data_or_current_hour=="all":
            start_date=start_date #"2010-01-01"
            end_date=datetime.now(timezone.utc).strftime('%Y-%m-%d')#current hour as str
        else:
            start_date=datetime.now(timezone.utc).strftime('%Y-%m-%d') #current hour as str
            end_date=datetime.now(timezone.utc).strftime('%Y-%m-%d') #current hour as str
        params = {
        "latitude": [18.5204, 43.5890],     # Pune, Mississauga
        "longitude": [73.8567, -79.6441],   # Pune, Mississauga
        "start_date": start_date, #can make this dynamic 
        "end_date": end_date,
        "hourly": ["temperature_2m", "rain", "wind_speed_10m", "pressure_msl"],
            }
        
        responses = openmeteo.weather_api(data_url, params=params)
        all_dfs = []
        locations = ["Pune", "Mississauga"]
        for idx, response in enumerate(responses):
            lat = response.Latitude()
            lon = response.Longitude()
            hourly = response.Hourly()
            temps = hourly.Variables(0).ValuesAsNumpy()
            rain = hourly.Variables(1).ValuesAsNumpy()
            wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
            pressure_msl = hourly.Variables(3).ValuesAsNumpy()
            

            time_index = pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end   = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq  = pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )    
            df = pd.DataFrame({
            "date":           time_index,
            "temperature_2m": temps,
            "rain": rain,
            "wind_speed_10m": wind_speed_10m,
            "pressure_msl": pressure_msl,
            "latitude":       lat,
            "longitude":      lon,
            "location":       locations[idx]
            })
            all_dfs.append(df)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        my_logger.info(f"{datetime.now()}:Data Pulled in for params {params}")
        combined_df.to_csv(f'./data/temp_data_{all_data_or_current_hour}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.csv')
        return f'./data/temp_data_{all_data_or_current_hour}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.csv'        
    except Exception as e:
        print(e)
        my_logger.info(f"{datetime.now()}:Error!:{e}")

if __name__=="__main__":
    pull_data("https://archive-api.open-meteo.com/v1/archive",'all',"2010-01-01") #pull data for the whole timeframe 