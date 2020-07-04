## Dashboard Script ##

# Note: After skipping the first tab 'Main Window', and going through the other tabs, to return to the 'Main Window' it may be necessary to reload the url link.#
# This due to heavy loading of mapgraph #

# Libraries

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
import seaborn as sn
import plotly.graph_objs as go
import os
import glob
import flask
import plotly.express as px
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# External CSS Template for Dash platform
externalTemplate = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Importing the LightGBM model
load = lgb.Booster(model_file='LightgbmEnergyPricePrediction_Project.txt')

# Energy Data
energyData = pd.read_csv('Data/energy_dataset.csv')
energyData = energyData.set_index('time', drop=True)

# Data with new index (datetime + city)
weatherData = pd.read_csv('Data/weather_data_update.csv')

# CompleteDataset model from
completeDataset = pd.read_csv('Data/complete_dataset.csv')
completeDataset = completeDataset.set_index('time')

# CompleteDataset_Energy_Study
completeDataset_energy_study = pd.read_csv('Data/completeDataset_energy_source.csv')

# Setting index
weatherData = weatherData.set_index('dt_iso')

# Listing dates and cities
cities = weatherData['city_name'].unique()
dates = weatherData.index.values.tolist()


# Create 2020 Test Set
presentYear = pd.DataFrame(columns=['time'])

# Function to create dates for 2020 test set
def dategenerator(start, end):
    delta = timedelta(hours=1)
    while start < end:
        yield start
        start += delta

# Defining new datetime
date_time = []

startDate = datetime(2020, 1, 1, 00, 00, 00)
endDate = datetime(2020, 12, 31, 23, 00, 00)
for i in dategenerator(startDate, endDate):
    date_time.append(i.strftime("%Y-%m-%d %H:%M:%S"))

## TPOT Model Prediction 2015-2018

# Define features and label
features = completeDataset.drop(['price actual_€/Mwh'], axis=1)
label = completeDataset['price actual_€/Mwh']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)

# Pipeline
exported_pipeline = make_pipeline(
        StackingEstimator(
            estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=1, min_samples_split=4,
                                          n_estimators=100)),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        LassoLarsCV(normalize=False)
    )

# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 54)
exported_pipeline.fit(X_train, y_train)
features['price €/Mwh'] = exported_pipeline.predict(features)

# LightGBM Model Prediction for 2020

presentYear['time'] = date_time
presentYear = presentYear.set_index('time')
presentYear['price €/Mwh'] = load.predict(presentYear.values)

# App definition
app = dash.Dash(__name__, external_stylesheets=externalTemplate)

# Using function for dataframe representation
def generateTable(dataframe, max_rows=15):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# General Main Dataset Statistics
genInfo = completeDataset.describe().reset_index(drop=False)
genInfo = genInfo.drop('index', axis=1)
genInfo = genInfo.rename(columns={'level_0':'Statistics'})

# General Information Energy Sources Statistics
sourceInfo = completeDataset_energy_study[['renewables_MWh', 'coal_oil_fossil_MWh', 'generation biomass_MWh']].describe().reset_index(drop=False)


# Defining Styles for Tab Sections

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '15px',
    'fontWeight': 'bold',
    'background': 'blue',
    'margin-right': 'auto',
}

tab_selected_style = {
    'borderTop': '15px solid #d6d6d6',
    'borderBottom': '15px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '15px',
    'margin-right': 'auto'
}

tab_selected_style_map = {
    'borderTop': '1px solid #d6d6d6',
    'fontColor': 'white',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'blue',
    'color': 'white',
    'padding': '15px',
    'margin-right': 'auto'
}

# Dropdown Type
dropdown_type = {
    "background-color": "white",
    "color": "blue",
    # "color": "#ffffff",
    # "fontColor": "white",
    # "font-color": "white",
    "width": "500px",
    "font-family": "sans-serif",
    "font-size": "large",
}

# App Architecture
app.layout = html.Div(style = {'color':'white', 'background-color':'rgb(60,60,60)', 'padding':'100'},
                      children=[html.H1('Energy Price Forecasting Service - Spain', style={'textAlign': 'center'}),
                      # Description
                      html.P('The goal of this project is to offer a broader range of users a national monitorization platform of electricity consumption and price levels, as well as additional features.', style={'textAlign': 'center'}),
                      #html.Label('', style={'textAlign': 'center'}),
                                # Tabs Section
                                dcc.Tabs(id="tabs", children=[
                                    # Main Window Tab
                                    dcc.Tab(label='Main Window', style=tab_style, selected_style= tab_selected_style,  children=[html.H4(children='City Weather Temperature Information',  style={'textAlign': 'center'}),
                                            dcc.Dropdown(id='menu',options=[{'label': i, 'value': i} for i in dates], value= '2015-01-01 00:00:00+01:00Valencia', style = dropdown_type),
                                            dcc.Graph(id='map', style=tab_selected_style_map),
                                            html.H2('Demand Forecasting', style={'textAlign': 'center'}),
                                            dcc.Graph(id='timeseries1', config={'displayModeBar': False}, animate=True, figure=px.line(energyData,
                                            x=energyData.index, y=energyData['total load actual']).update_layout( {'plot_bgcolor': 'rgba(5, 0, 0, 5)', 'paper_bgcolor': 'rgba(5, 0, 0, 5)'})),
                                            html.H2('Current Price Forecasting', style={'textAlign': 'center'}),
                                            dcc.Graph(id='timeseries2', config={'displayModeBar': False}, animate=True, figure=px.line(energyData, x=energyData.index, y=energyData['price actual']).update_layout({'plot_bgcolor': 'rgba(5, 0, 0, 5)', 'paper_bgcolor': 'rgba(5, 0, 0, 5)'}))
                                            ]
                                        ),
                                    # Data Description Tab
                                    dcc.Tab(label='Data Description', style=tab_style, selected_style=tab_selected_style, children=[html.H4(children='Data Resume', style={'textAlign': 'center'}),generateTable(completeDataset.head(10)), html.H4(children='General Data Statistics', style={'textAlign': 'center'}), generateTable(genInfo)]),
                                    # TPOT Model 2015-2018 Tab
                                    dcc.Tab(label='TPOT Price Prediction', style=tab_style, selected_style= tab_selected_style, children=[html.H4(children='Model TPOT Prediction',  style={'textAlign': 'center'}),
                                            dcc.Graph(id='result1', figure = {'data': [ {'x': features.index, 'y': features['price €/Mwh'], 'type': 'line', 'background-color':'rgb(0,0,0)'}],  'layout': {
                                                      'title': 'TPOT Model Prediction 2015-2018',
                                                      'xaxis': {
                                                          'title': 'Day Time',

                                                      },
                                                      'yaxis': {
                                                          'title': 'Price Forecast €/MWh',
                                                      },
                                                    },
                                                 }
                                                ),


                                              ]
                                            ),
                                    # Energy Source Analysis Tab
                                    dcc.Tab(label='Energy Source Contribution', style=tab_style, selected_style=tab_selected_style, children=[html.H4(children='Energy Source Contributions', style={'textAlign': 'center'}),generateTable(sourceInfo),
                                            dcc.Graph(id='Plots', figure={ 'data':[ {'x': completeDataset.index, 'y': completeDataset['renewables_MWh'], 'type':'line', 'name':'Renewables' }, {'x': completeDataset.index, 'y': completeDataset['coal_oil_fossil_MWh'], 'type':'line', 'name':'Coal Oil Fossil'}, {'x': completeDataset.index, 'y': completeDataset['generation biomass_MWh'], 'type':'line', 'name':'Biomass'}], 'layout': {
                                                      'title': 'Energy Source Contribution 2015-2018',
                                                      'xaxis': {
                                                          'title': 'Day Time',

                                                      },
                                                      'yaxis': {
                                                          'title': 'Demand Forecast MW/h',
                                                      },
                                                    },
                                                  },
                                                ),
                                              ]
                                            ),
                                    # LightGBM Model Tab
                                    dcc.Tab(label='LightGBM Price Prediction', style=tab_style, selected_style= tab_selected_style, children=[html.H4(children='Model LightGBM Prediction',  style={'textAlign': 'center'}),
                                            dcc.Graph(id='result2', figure = {'data': [ {'x': presentYear.index, 'y': presentYear['price €/Mwh'], 'type': 'line', 'background-color':'rgb(0,0,0)'}],  'layout': {
                                                      'title': 'LightGBM Model Prediction 2020',
                                                      'xaxis': {
                                                          'title': 'Day Time',

                                                      },
                                                      'yaxis': {
                                                          'title': 'Price Forecast €/MWh',
                                                      },
                                                  },
                                                }
                                            ),


                                         ]
                                    ),
                                ]
                            )
                        ]
                      )




# Callback functions
@app.callback(
    [Output('map', 'figure'),
     Output('map', 'config'),
     ],
    [Input('menu', 'value')])

# Upgrade map
def update_graph(value):
    return update_map_callback(value)

# Map
def update_map_callback(date):
    map_figure = {
        'data': [
            go.Scattermapbox(
                lat=weatherData['Latitude'],
                lon=weatherData['Longitude'],
                mode='markers',
                marker=dict(
                    size=13,
                ),
                text= weatherData['temp'][date]
            )
        ],
        'layout': go.Layout(
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                accesstoken='pk.eyJ1IjoidG9kZGthcmluIiwiYSI6Ik1aSndibmcifQ.hwkbjcZevafx2ApulodXaw',
                center=dict(
                    lat=40,
                    lon=1
                ),
                zoom=3
            )
        )}

    map_config = dict(scrollZoom=True)

    return map_figure, map_config


# Running App
if __name__ == '__main__':
    app.run_server(debug=True)