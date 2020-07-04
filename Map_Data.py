# Libraries
import numpy as np
import pandas as pd


weatherData = pd.read_csv('a.csv')

# Set new index
for i in range(len(weatherData)):
    if weatherData['city_name'][i] == 'Seville':
        weatherData['dt_iso'][i] = weatherData['dt_iso'][i] + 'Seville'
    elif weatherData['city_name'][i] == ' Barcelona':
        weatherData['dt_iso'][i] = weatherData['dt_iso'][i] + 'Barcelona'
    elif weatherData['city_name'][i] == 'Bilbao':
        weatherData['dt_iso'][i] = weatherData['dt_iso'][i] + 'Bilbao'
    elif weatherData['city_name'][i] == 'Valencia':
        weatherData['dt_iso'][i] = weatherData['dt_iso'][i] + 'Valencia'
    elif weatherData['city_name'][i] == 'Madrid':
        weatherData['dt_iso'][i] = weatherData['dt_iso'][i] + 'Madrid'

print(weatherData)

weatherData.to_csv('weather_data_update.csv')