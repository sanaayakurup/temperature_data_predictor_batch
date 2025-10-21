"""
This file will read in data from the hitorical api. 
We will use this data to train the model
"""
print("ji")
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from utils import setup_logger
import logging
import datetime
from datetime import date

#https://open-meteo.com/en/features#available_apis

my_logger = setup_logger('my_temperature_logger', 'read_data.log', level=logging.DEBUG)
my_logger.info(f"{datetime.datetime.now()}:Initialising the API connection")

try: 
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
    "latitude": [18.5204, 43.5890],     # Pune, Mississauga
    "longitude": [73.8567, -79.6441],   # Pune, Mississauga
    "start_date": "2010-01-01", #can make this dynamic 
    "end_date": "2019-12-31",
    "hourly": "temperature_2m",
        }
    responses = openmeteo.weather_api(url, params=params)
    all_dfs = []
    locations = ["Pune", "Mississauga"]
    for idx, response in enumerate(responses):
        lat = response.Latitude()
        lon = response.Longitude()
        hourly = response.Hourly()
        temps = hourly.Variables(0).ValuesAsNumpy()
        time_index = pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end   = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq  = pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )    
        df = pd.DataFrame({
        "date":           time_index,
        "temperature_2m": temps,
        "latitude":       lat,
        "longitude":      lon,
        "location":       locations[idx]
        })
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    my_logger.info(f"{datetime.datetime.now()}:Data Pulled in for params {params}")
    combined_df.to_csv(f'temp_data_{date.today()}.csv')        
except Exception as e:
    print(e)
    my_logger.info(f"{datetime.datetime.now()}:Error!:{e}")

