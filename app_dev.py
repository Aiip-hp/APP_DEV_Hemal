"""Dash is a Python framework for building web applications. It is built on top of Flask, Plotly.js, React
 and React Js., enabling you to build dashboards & applications using pure Python"""
 
# Creating an Interactive Data app using Plotly’s Dash
""" Resources; https://dash.plotly.com/external-resources
https://towardsdatascience.com/creating-an-interactive-data-app-using-plotlys-dash-356428b4699c"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os
import matplotlib.pyplot as pyplot
import pickle
from sklearn import model_selection

# %matplotlib inline

# Solar annual generation data;
solar_annual_gen=pd.read_csv('solar_generation_data.csv') # load solar geeneration data file on to a pandas data frame

# Wind annual generation data
wind_annual_gen=pd.read_csv('wind_generation_data.csv') # load Wind geeneration data file on to a pandas data frame
# seems the  data is for a little more than an year (366days) and therefore last row shall be dropped
wind_annual_gen=wind_annual_gen[0:365]

# Additional data preparation
# Mapping month string values to numeric
month_num={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,
        'Nov':11,'Dec':12}

solar_annual_gen['Month_number']=solar_annual_gen.iloc[:,0].map(month_num)

# mapping of Month_number and Day to wind_annual_gen dataframe from solar data
wind_day_month=solar_annual_gen[['Day','Month_number']]

# Merge the data frames; wind_day_month to wind_annual_gen
wind_annual_gen=pd.merge(wind_day_month, wind_annual_gen, right_index=True, left_index=True)

# Remove the special character "°" and Convert Hi and low temp into floats
solar_annual_gen_temp=solar_annual_gen[["Temp Hi","Temp Low"]].replace('\°','',regex=True).astype(float)
solar_annual_gen_temp.columns = ['Temp Hi(float)','Temp Low(Float)']

# Merging high and low temp to original solar generation data frame
solar_annual_gen = pd.merge(solar_annual_gen, solar_annual_gen_temp, right_index=True, left_index=True)


# Impute missing values with median i.e. Rainfall in mm with 53 missing value can be replaced with the median value (0) 
# imputing missing values with median is preffered to reduce bias as a result of skewness or outliers
solar_annual_gen["Rainfall in mm" ].fillna(solar_annual_gen["Rainfall in mm" ].median(), inplace=True)


# Split dataset into predictor and target matrices & Create model and save using pickling
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

solar_annual_gen_numeric = solar_annual_gen.select_dtypes(include=numerics) # Include only numeric data types

"""Solar Generation data columns for use in model;
 Index(['Day', 'Solar', 'Cloud Cover Percentage', 'Rainfall in mm',
       'Power Generated in MW', 'Month_number', 'Temp Hi(float)',
       'Temp Low(Float)'],
      dtype='object')

Wind Generation data columns for use in model;
 Index(['Day', 'Month_number', 'wind speed', 'direction', 'Power Output'], dtype='object')"""
 
 # Split dataset into predictor and target matrices

# Values of target
y = solar_annual_gen_numeric['Power Generated in MW'].values
y1= wind_annual_gen['Power Output'].values

# Values of attributes
# For Solar, Drop the target value and 'Rainfall in mm'(no data from the weather forecast data)
# For Wind, Drop the target value
solar_annual_gen_X = solar_annual_gen_numeric.drop(['Power Generated in MW','Rainfall in mm'], axis=1)
wind_annual_gen_X = wind_annual_gen.drop(['Power Output'], axis=1)
X = solar_annual_gen_X.values
X1 = wind_annual_gen_X.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)# Use test size of 25% for better
# performance

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=0)# Use test size of 25% for better
# performance
 
# Fit the model on training set
solarmodel = RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=-1)
solarmodel.fit(X_train, y_train)
windmodel = RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=-1)
windmodel.fit(X1_train, y1_train)

# Save Model Using Pickle
import pickle
pickle.dump(solarmodel, open('solar_model.pkl','wb'))
pickle.dump(windmodel, open('wind_model.pkl','wb'))

'''The following function is used for creating the final data frames for the solar and wind weather forecast data.
From the openweathermap website, the urls for solar and wind are inserted to read the forecast data in json ,convert into pandas data frames, 
drop unwanted data columns  and format columns to match the format of the model training data sets for solar and wind'''

# def solar_wind():
    
# Import libraries
import requests
import pandas as pd
    
# Using "https://openweathermap.org/api/one-call-api"
# To make the api call, the following format is used as per the websites documentaion;
# https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={part}&appid={API key}
"""Parameters; lat, lon required	Geographical coordinates (latitude, longitude),
appid	required	Your unique API key (you can always find it on your account page under the "API key" tab)
exclude	optional	By using this parameter you can exclude some parts of the weather data from the API response.
It should be a comma-delimited list (without spaces).Available values:current,minutely,hourly,daily,alerts"""

# urls for solar and wind generation data forecast
solar_url = "https://api.openweathermap.org/data/2.5/onecall?lat=-19.461907&lon=142.110216&exclude=[current,minutely,hourly]&appid=c8b03af1f5efccda17ff89c4e0df6b0b"
wind_url="https://api.openweathermap.org/data/2.5/onecall?lat=53.556563&lon=8.598084&exclude=[current,minutely,hourly]&appid=686a8268d2d60adfa1efd1b0f3d7ffe5"

# get daily weather forcast data in json format
jsonData_solar = requests.get(solar_url).json()
jsonData_wind = requests.get(wind_url).json()

# convert the data-interchange format, json, to a pandas data frame for ease of manipulation & use in the regression model
weather_data_solar = pd.DataFrame(jsonData_solar['daily'])
weather_data_wind = pd.DataFrame(jsonData_wind['daily'])

"""for solar Split temp sub columns data (i.e. 'day','min','max', 'eve', 'morn')  into seperate usable columns and merge it to the 
original weather_data_solar frame"""
temp_sub_data=weather_data_solar['temp'].apply(pd.Series)
weather_solar=pd.merge(weather_data_solar, temp_sub_data, right_index=True, left_index=True)

# Extract day and month columns from the weather data similar to the training data
weather_solar['Day']=pd.to_datetime(weather_solar['dt'],unit = 's').dt.day
weather_solar['Month']=pd.to_datetime(weather_solar['dt'],unit = 's').dt.month
weather_data_wind['Day']=pd.to_datetime(weather_data_wind['dt'],unit = 's').dt.day
weather_data_wind['Month']=pd.to_datetime(weather_data_wind['dt'],unit = 's').dt.month

# Reduce the 8day forecast to 7 days as required;
weather_solar=weather_solar.iloc[0:7,]
weather_wind=weather_data_wind.iloc[0:7,]

"""Solar forecast data columns;
Index(['dt', 'sunrise', 'sunset', 'temp', 'feels_like', 'pressure', 'humidity',
   'dew_point', 'wind_speed', 'wind_deg', 'weather', 'clouds', 'pop',
   'uvi', 'day', 'min', 'max', 'night', 'eve', 'morn', 'Day', 'Month'],
   dtype='object')
   
   Wind forecast data columns;
   Index(['dt', 'sunrise', 'sunset', 'temp', 'feels_like', 'pressure', 'humidity',
   'dew_point', 'wind_speed', 'wind_deg', 'weather', 'clouds', 'pop',
   'snow', 'uvi', 'rain', 'Day', 'Month'],
   dtype='object')"""
   
"""Required columns for solar attributes (X)=['Day', 'Solar', 'Cloud Cover Percentage', 
'Month_number','Temp Hi(float)', 'Temp Low(Float)']"""

"""Required columns for wind attributes (X1)=['Day', 'Month_number', 'wind speed', 'direction']"""

weather_solar=weather_solar[['Day','uvi','clouds','Month','max','min']]
weather_wind=weather_wind[['Day', 'Month', 'wind_speed', 'wind_deg']]

# Rename columns to match Training data columns
solar=weather_solar.rename(columns={"uvi":"Solar","clouds":"Cloud Cover Percentage","Month": "Month_number",
                            "max": "Temp Hi(float)","min": "Temp Low(Float)"})
wind=weather_wind.rename(columns={"Month": "Month_number","wind_speed": "wind speed","wind_deg": "direction"})

# return solar,wind
	
# import libraries
import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.graph_objs as go
# from api_solar_wind import solar_wind
import pandas as pd
import pickle

# Creating an Interactive Data app using Plotly’s Dash

""" Resources; https://dash.plotly.com/external-resources
https://towardsdatascience.com/creating-an-interactive-data-app-using-plotlys-dash-356428b4699c"""

# Styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

# Load Saved Model i.e. the solar and wind model in .pkl format
model = pickle.load(open('solar_model.pkl','rb'))
model1 = pickle.load(open('wind_model.pkl','rb'))

# HTML
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Upload Maintenance Schedule Files for Solar & Wind Farms')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),dcc.Graph(id='Mygraph'),
    dcc.Graph(id='Mygraph1'),
],style={'height':'20px','width':'60%','margin-left':'auto',
'margin-right':'auto','margin-top': 'auto','margin-bottom': 'auto'})

# Function for Parsing uploaded maintenance schedule files
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assuming that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assuming that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'Wrong Filetype For Maintenance Schedule'
        ])
    return df

# Combining the two callback for solar and wind;
""" Callback for solar generation for uploading solar farm maintenance schedule and plotting predicted wind
power output"""
# Reference on plots using plotly express[ https://plotly.com/python/line-charts/]

@app.callback(Output('Mygraph', 'figure'),
            [
                Input('upload-data', 'contents'),
                Input('upload-data', 'filename')
            ])
# Create Function for Solar Predictions and output visualization
def update_graph(contents, filename):
    fig = {
        'layout': go.Layout(
            plot_bgcolor=colors["graphBackground"],
            paper_bgcolor=colors["graphBackground"])
    }

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        
        """retrieve data from the solar_wind()function, which loads 7 day forecasts from the
        weather api in the desired data frame column formats for solar and wind """
        solar_weather_forecast=solar
        # print(solar_weather_forecast,wind_weather_forecast) # confirmed successfull loading of the next 7 days weather forecast data

        # predict solar output
        solar_predicted_output = model.predict(solar_weather_forecast) 
        # print(solar_predicted_output) # output confirmed

        # merge the predicted power output with the forcast data
        solar_weather_forecast["Predicted power output(MW)"]=pd.Series(solar_predicted_output)
        # print(solar_weather_forecast) # confirmed

        # rename "Date Of Month" to "Day" for maintenance schedule
#         solar_monthly_schedule=pd.read_csv('solar_farm.csv', skiprows=1)
        
        df.to_csv('sol_maint.csv',index=False)
        df=pd.read_csv('sol_maint.csv',skiprows=1)
        maint=df.rename(columns={"Date Of Month":"Day"})
       
        # print(maint) # confirmed

        """ map the % capacity available from the maintenance schedule to the solar forecast & predicted power dataframe 
        with common column="Day" """
        solar_final=pd.merge(solar_weather_forecast, maint,how='left')
        # print(df_sol_final) # confirmed

        # fill missing values with 100, meaning available capacity when there is no planned maintenace is 100% of the predicted output
        solar_final=solar_final.fillna(100)
        # print(solar_final) # confirmed

        # Derate the predicted output by the % capacity available
        solar_final['Predicted power output(MW)']=solar_final['Predicted power output(MW)']*(solar_final['Capacity Available']/100)
        # print(solar_final) # confirmed

        # Plot Line graph using plotly express      
        fig = px.line(solar_final, x="Day", y="Predicted power output(MW)",
                      title='7 day Predicted Solar power output(MW)',labels={"Day": "Day of the month"})
#         fig.show()

        dcc.Graph(figure=fig)
 
    return fig        

""" Callback for wind generation for uploading wind farm maintenance schedule and plotting predicted
wind power output"""
# Reference on plots using plotly express[ https://plotly.com/python/line-charts/]
@app.callback(Output('Mygraph1', 'figure'),
            [
                Input('upload-data', 'contents'),
                Input('upload-data', 'filename')
            ])
def update_graph1(contents, filename):
    fig1 = {
        'layout': go.Layout(
            plot_bgcolor=colors["graphBackground"],
            paper_bgcolor=colors["graphBackground"])
    }

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        
        """retrieve data from the solar_wind()function, which loads 7 day forecasts from the 
        weather api in the desired data frame column formats for solar and wind """
        wind_weather_forecast=wind

        # predict wind output
        wind_predicted_output = model1.predict(wind_weather_forecast) 
        
        # merge the predicted power output with the forcast data
        wind_weather_forecast["Predicted power output(MW)"]=pd.Series(wind_predicted_output)
        
        # rename "Date Of Month" to "Day" for maintenance schedule
#         wind_monthly_schedule=pd.read_csv('wind_farm.csv', skiprows=1)


        df.to_csv('wind_maint.csv',index=False)
        df=pd.read_csv('wind_maint.csv',skiprows=1) 
        maint1=df.rename(columns={"Date Of Month":"Day"})
        # print(maint1) # confirmed

        """ map the % capacity available from the maintenance schedule to the wind forecast & predicted power dataframe 
        with common column="Day" """
        wind_final=pd.merge(wind_weather_forecast, maint1,how='left')
        # print(wind_final) # confirmed

        # fill missing values with 100, meaning available capacity when there is no planned maintenace is 100% of the predicted output
        wind_final=wind_final.fillna(100)
        # print(wind_final) # confirmed

        # Derate the predicted output by the % capacity available
        wind_final['Predicted power output(MW)']=wind_final['Predicted power output(MW)']*(wind_final['Capacity Available']/100)
        # print(wind_final) # confirmed

        # Plot Line graph using plotly express
        fig1 = px.line(wind_final, x="Day", y="Predicted power output(MW)",
                      title='7 day Predicted Wind power output(MW)',labels={"Day": "Day of the month"})
#         fig.show()

        dcc.Graph(figure=fig1)
 
    return fig1        

if __name__ == '__main__':
    app.run_server(debug=True)
#    app.run_server(debug=False)


