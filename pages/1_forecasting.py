import pickle

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import tensorflow as tf
from datetime import timedelta

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import keras.initializers
from keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l1, l2

from model_functions import SimpleRNN_, GRU_, LSTM_, get_lagged_features, plot_forecast
import pickle

# allows reading of keras file. See issue:
# https://discuss.streamlit.io/t/attributeerror-thread-local-object-has-no-attribute-value/574/3

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

# Opens a css file that changes the font style of the app to Montserrat
with open('pages/styles.css') as css: 
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

##############################################################

# Open 2-year models

with open("pages/models_/2YR_models_rnn.pkl", "rb") as file:
     data_2yr_rnn = pickle.load(file)

with open("pages/models_/2YR_models_gru.pkl", "rb") as file:
     data_2yr_gru = pickle.load(file)

with open("pages/models_/2YR_models_lstm.pkl", "rb") as file:
     data_2yr_lstm = pickle.load(file)

data_2yr = {**data_2yr_rnn, **data_2yr_gru, **data_2yr_lstm}

# Open 10-year models

with open("pages/models/10YR_models_rnn.pkl", "rb") as file:
     data_10yr_rnn = pickle.load(file)

with open("pages/models/10YR_models_gru.pkl", "rb") as file:
     data_10yr_gru = pickle.load(file)

with open("pages/models/10YR_models_lstm.pkl", "rb") as file:
     data_10yr_lstm = pickle.load(file)

data_10yr = {**data_10yr_rnn, **data_10yr_gru, **data_10yr_lstm}

data_models = {'2yr': data_2yr, '10yr': data_10yr}


#########################################################

train_mean = data_2yr['rnn']['scaler_train_mean']
train_scale = data_2yr['rnn']['scaler_train_std']

st.write("### Upload data here:")

# Take input data from user
uploaded_csv_2yr = st.file_uploader("Input .csv file of 2-year yields here:", type="csv")

uploaded_csv_10yr = st.file_uploader("Input .csv file of 10-year yields here:", type="csv")

# get the models

model_rnn = data_2yr_rnn['rnn']['model']
model_gru = data_2yr_gru['gru']['model']
model_lstm = data_2yr_lstm['lstm']['model']

# get the transformer

scaler = StandardScaler()
# scaler.mean_ = test_mean
# scaler.scale_ = test_scale


# transform the input

if uploaded_csv_2yr is not None:
    df_2yr = pd.read_csv(uploaded_csv_2yr)
    df_2yr['date'] = pd.to_datetime(df_2yr['date'], infer_datetime_format=True)
    df_2yr.index = pd.to_datetime(df_2yr['date'], infer_datetime_format=True)
    df_2yr.drop(columns=['date'], inplace=True)
    
    fitted_scaler = scaler.fit(np.array(df_2yr['yield']).reshape(-1,1))
    
    df_2yr_scaled_ = fitted_scaler.transform(np.array(df_2yr['yield']).reshape(-1,1))
    for_input_2yr = df_2yr_scaled_.reshape(1, len(df_2yr), 1)

    models = ['rnn', 'gru', 'lstm']

    rnn_pred = fitted_scaler.inverse_transform(data_2yr['rnn']['model'].predict(for_input_2yr))
    gru_pred = fitted_scaler.inverse_transform(data_2yr['gru']['model'].predict(for_input_2yr))
    lstm_pred = fitted_scaler.inverse_transform(data_2yr['lstm']['model'].predict(for_input_2yr))

    # st.write(f"RNN predictions: {rnn_pred}")
    # st.write(f"GRU predictions: {gru_pred}")
    # st.write(f"LSTM predictions: {lstm_pred}")

    # predictions = [rnn_pred, gru_pred, lstm_pred]


    st.write("### Forecast plot (2yr UST):")
    plot_forecast(input_dates=df_2yr.index,
                  input_data=df_2yr,
                  forecast_rnn=rnn_pred,
                  forecast_gru=gru_pred,
                  forecast_lstm=lstm_pred,
                  dataset_="2yr")
     
    

if uploaded_csv_10yr is not None:
    df_10yr = pd.read_csv(uploaded_csv_10yr)


