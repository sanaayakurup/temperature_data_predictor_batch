
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
data=pd.read_csv('temp_data_2025-10-21.csv').drop(columns='Unnamed: 0')
data.head()
data.date=pd.to_datetime(data.date)
data.info()
#for pune and Sauga, plot temps
# #%% 
for i in data.location:
    filt=data.query(f"location=='{i}'")
    px.line(filt,x='date',y='temperature_2m')
