from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV


import keras.initializers
from keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import plotly.graph_objects as go

import io
from datetime import date

unroll_setting = False # setting to false saves memory consumption

def SimpleRNN_(n_units, l1_reg, seed, x_train_, n_steps_ahead: int):
  model = Sequential()
  model.add(
     SimpleRNN(n_units, 
               activation='tanh', 
               kernel_initializer=keras.initializers.glorot_uniform(seed), 
               bias_initializer=keras.initializers.glorot_uniform(seed), 
               recurrent_initializer=keras.initializers.orthogonal(seed), 
               kernel_regularizer=l1(l1_reg), 
               input_shape=(x_train_.shape[1], x_train_.shape[-1]), 
               unroll=unroll_setting, stateful=False)
               ) 
  model.add(
     Dense(n_steps_ahead, 
           kernel_initializer=keras.initializers.glorot_uniform(seed), 
           bias_initializer=keras.initializers.glorot_uniform(seed), 
           kernel_regularizer=l1(l1_reg))
           )
  model.compile(
     loss='mean_squared_error', 
     optimizer='adam'
     )
  return model

def GRU_(n_units, l1_reg, seed, x_train_, n_steps_ahead: int):
  model = Sequential()
  model.add(
     GRU(n_units, 
               activation='tanh', 
               kernel_initializer=keras.initializers.glorot_uniform(seed), 
               bias_initializer=keras.initializers.glorot_uniform(seed), 
               recurrent_initializer=keras.initializers.orthogonal(seed), 
               kernel_regularizer=l1(l1_reg), 
               input_shape=(x_train_.shape[1], x_train_.shape[-1]), 
               unroll=unroll_setting, stateful=False)
               )
  model.add(
     Dense(n_steps_ahead, 
           kernel_initializer=keras.initializers.glorot_uniform(seed), 
           bias_initializer=keras.initializers.glorot_uniform(seed), 
           kernel_regularizer=l1(l1_reg))
           )
  model.compile(
     loss='mean_squared_error', 
     optimizer='adam'
     )
  return model

def LSTM_(n_units, l1_reg, seed, x_train_, n_steps_ahead: int):
  model = Sequential()
  model.add(
     GRU(n_units, 
        activation='tanh', 
        kernel_initializer=keras.initializers.glorot_uniform(seed), 
        bias_initializer=keras.initializers.glorot_uniform(seed), 
        recurrent_initializer=keras.initializers.orthogonal(seed), 
        kernel_regularizer=l1(l1_reg), 
        input_shape=(x_train_.shape[1], x_train_.shape[-1]),
        unroll=unroll_setting, stateful=False)
        ) 
  model.add(
     Dense(n_steps_ahead, 
           kernel_initializer=keras.initializers.glorot_uniform(seed), 
           bias_initializer=keras.initializers.glorot_uniform(seed), 
           kernel_regularizer=l1(l1_reg))
           )
  model.compile(
     loss='mean_squared_error', 
     optimizer='adam'
     )
  return model

def get_lagged_features(df, n_steps, n_steps_ahead):
    """
    df: pandas DataFrame of time series to be lagged
    n_steps: number of lags, i.e. sequence length
    n_steps_ahead: forecasting horizon
    """
    lag_list = []
    
    for lag in range(n_steps + n_steps_ahead - 1, n_steps_ahead - 1, -1):
        lag_list.append(df.shift(lag))
    lag_array = np.dstack([i[n_steps+n_steps_ahead-1:] for i in lag_list])
    # We swap the last two dimensions so each slice along the first dimension
    # is the same shape as the corresponding segment of the input time series 
    lag_array = np.swapaxes(lag_array, 1, -1)
    return lag_array

def plot_forecast(input_dates, input_data, forecast_rnn, forecast_gru, forecast_lstm, dataset_):
    
    arr_rnn = np.insert(forecast_rnn, 0, input_data[-1:].values.item())
    arr_gru = np.insert(forecast_gru, 0, input_data[-1:].values.item())
    arr_lstm = np.insert(forecast_lstm, 0, input_data[-1:].values.item())
    
    # Plot forecast data from RNN, GRU, and LSTM (next 4 days)
    forecast_dates = pd.date_range(start=input_dates[-2] + pd.Timedelta(days=1), periods=5, freq='D')
    
    df = pd.DataFrame({
        'rnn': arr_rnn,
        'gru': arr_gru,
        'lstm': arr_lstm
    },
        index=forecast_dates)
    
    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Plot actual data (10 past days)
    plt.plot(input_dates, input_data, label="Input Data (15 Days)", color='black', linewidth=2)

    # RNN Forecast (Red)
    plt.plot(forecast_dates, df['rnn'], label="RNN Forecast", color='blue', linestyle='--', linewidth=2)

    # GRU Forecast (Green)
    plt.plot(forecast_dates, df['gru'], label="GRU Forecast", color='green', linestyle='--', linewidth=2)

    # LSTM Forecast (Purple)
    plt.plot(forecast_dates, df['lstm'], label="LSTM Forecast", color='red', linestyle='--', linewidth=2)

    # Adding title and labels
   #  plt.title("Time Series Forecasting with RNN, GRU, and LSTM", fontsize=16)
    plt.xlabel("Date", fontsize=24)
    plt.ylabel("Yield", fontsize=24)

    # Formatting the x-axis to display dates
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show every day
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as date (YYYY-MM-DD)
    plt.xticks(rotation=45)  # Rotate labels for readability

    # Displaying the legend
    plt.legend()
    

    # Adding grid for better visibility
    plt.grid(True)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Show the plot
    st.pyplot(plt)

    # Create a buffer to save the DataFrame to an Excel file
    buffer = io.StringIO()
    
   #  with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
   #      df.to_excel(writer, index=False, sheet_name='Forecasts')
   #      writer.save()

    df.drop(index=df.index[0], axis=0, inplace=True)

   # Write the DataFrame to the CSV buffer
    df.to_csv(buffer, index=True)

   # Reset the buffer position to the start
    buffer.seek(0)

    # Create the download button
    st.download_button(
        label=f"Download {dataset_} yields as csv",
        data=buffer.getvalue(),
        file_name=f"forecast_data_{date.today()}.csv",
        mime="text/csv"
    )

    # Create a buffer to save the DataFrame to an Excel file
    buffer_plt = io.BytesIO()
    
   #  with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
   #      df.to_excel(writer, index=False, sheet_name='Forecasts')
   #      writer.save()
    plt.savefig(buffer_plt, format="png")
    # Reset the buffer position to the start
    buffer_plt.seek(0)

# Add a download button
    st.download_button(label=f"Download {dataset_} forecast plot",
                       data=buffer_plt,
                       file_name=f"forecast_plot_{date.today()}.png",
                       mime="image/png"
                       )