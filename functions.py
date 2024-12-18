import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
from datetime import timedelta

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


import keras.initializers
from keras.layers import Dense, LSTM, GRU, SimpleRNN
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

import pickle
import gc

unroll_setting = False

# convert history into inputs and outputs
def reformat_to_arrays(data, n_steps, n_steps_ahead):
    X, y = list(), list()
    in_start = 0
    
    for _ in range(len(data)):
        in_end = in_start + n_steps
        out_end = in_end + n_steps_ahead
        
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
        
    return np.array(X), np.array(y)

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


# This function splits the data into training, validation, and test sets
# df = dataframe to split
# train_pct = The percentage of the data we want to be treated as training set. (e.g. if we want 60%, then training_size = 0.6)
# val_pct = similar to training_size, but for validation set
def split_data(df, train_pct, val_pct):

    # Check if the sum of train_pct and val_pct exceeds 1 
    if train_pct + val_pct > 1:
        raise ValueError(f"The sum of train_pct and val_pct should not exceed 1.")

     
    # Define the split ratios
    train_size = int(len(df) * train_pct) 
    val_size = int(len(df) * val_pct)

    # We get the train size by subtracting the train_size and val_size
    test_size = len(df) - train_size - val_size 

    # Split the data
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    # returns numpy arrays of train, val, and test
    return train_data, val_data, test_data
	
# Scales the given data using StandardScaler()
# Also saves the mean and standard deviation computed from fitting the StandardScaler() with the given data.
# We save the mean and standard deviation because the input for models are scaled data, therefore we will 
# scale the input data before inputting in the models. Outputs are also scaled and we inverse_scale them to
# get the proper value.
def scale_the_data(data):
    scaler = StandardScaler()

    scaler_fit = scaler.fit(data)
    mean = scaler_fit.mean_
    std = scaler_fit.scale_

    transformed_data = scaler_fit.transform(data)

    return transformed_data, mean, std, scaler_fit

def cross_val(params, batch_size, max_epochs, x_train_, y_train_, es, n_steps_ahead):

    n_units = [5, 10, 20, 25, 30]
    l1_reg = [0.001, 0.01, 0.1]
    
    
    # A dictionary containing a list of values to be iterated through
    # for each parameter of the model included in the search
    param_grid = {'n_units': n_units, 'l1_reg': l1_reg}
    
    # In the kth split, TimeSeriesSplit returns first k folds 
    # as training set and the (k+1)th fold as test set.
    tscv = TimeSeriesSplit(n_splits = 4)
    
    # A grid search is performed for each of the models, and the parameter set which
    # performs best over all the cross-validation splits is saved in the `params` dictionary
    for key in params.keys():
        print('Performing cross-validation. Model:', key)
        
        # add start time
        start_time_cv = time.time()
        
        if key == 'lstm':
            model = KerasRegressor(
            build_fn=lambda n_units, l1_reg: LSTM_(n_units=n_units, l1_reg=l1_reg, seed=0, x_train_=x_train_, n_steps_ahead=n_steps_ahead),
            epochs=max_epochs, 
            batch_size=batch_size,
            verbose=2
        )
        elif key == 'rnn':
            model = KerasRegressor(
            build_fn=lambda n_units, l1_reg: SimpleRNN_(n_units=n_units, l1_reg=l1_reg, seed=0, x_train_=x_train_, n_steps_ahead=n_steps_ahead),
            epochs=max_epochs, 
            batch_size=batch_size,
            verbose=2
        )
        elif key == 'gru':
            model = KerasRegressor(
            build_fn=lambda n_units, l1_reg: GRU_(n_units=n_units, l1_reg=l1_reg, seed=0, x_train_=x_train_, n_steps_ahead=n_steps_ahead),
            epochs=max_epochs, 
            batch_size=batch_size,
            verbose=2
        )
   
        
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=tscv,
            verbose=0
        )
        
        grid_result = grid.fit(
            x_train_,
            y_train_,
            callbacks=[es]
        )
        
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
        #add end time
        
        end_time_cv = time.time()
        
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params_ = grid_result.cv_results_['params']
        for mean, stdev, param_ in zip(means, stds, params_):
            print("%f (%f) with %r" % (mean, stdev, param_))
            
        params[key]['H'] = grid_result.best_params_['n_units']
        params[key]['l1_reg']= grid_result.best_params_['l1_reg']
        params[key]['cv_time'] = end_time_cv - start_time_cv


        del model, grid, grid_result, means, stds, params_
        gc.collect()


# reads the csv file for a given tenor ("2YR" or "10YR")
def read_data(tenor: str):

    # Validate the tenor input to ensure it's a valid format
    if not isinstance(tenor, str):
        raise ValueError("Tenor must be a string.")
    if tenor.upper() not in ['2YR', '10YR']:  # you can adjust this based on your valid options
        raise ValueError("Invalid tenor. Only '2YR' or '10YR' are allowed for now.")

    df = pd.read_csv(
        f'./data/bond_yields - USGG{tenor.upper()}.csv',
        index_col=0 # will read the first column as the index
        )

    df.index = pd.to_datetime(
        df.index,
        infer_datetime_format=True
        )
    
    df = df.loc['2005-01-01':'2024-08-31'] # get only from year 2005

    return df

def plotPACF(df: pd.DataFrame, filename: str):

    use_features = "yield"

    adf, p, usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(df[use_features])
    adf_results_string = 'ADF: {}\np-value: {},\nN: {}, \ncritical values: {}'
    # print(adf_results_string.format(adf, p, nobs, cvs))

    pacf = sm.tsa.stattools.pacf(df[use_features], nlags=usedlag)

    T = len(df[use_features])

    z_score = 2.58 # 99% confidence interval

    plt.plot(pacf, label='pacf')
    plt.plot([z_score/np.sqrt(T)]*30, label='99% confidence interval (upper)')
    plt.plot([-z_score/np.sqrt(T)]*30, label='99% confidence interval (lower)')
    plt.xlabel('number of lags')
    plt.legend()
    plt.show()
    plt.savefig(f"figures/{filename}.png", dpi=1000)


def train_model(dataset_, model_, n_steps_ahead, params, batch_size, max_epochs, es, train_val_test_dict):

    for key in params.keys():
        tf.random.set_seed(0)
        print('Training', key, 'model')

        X_train_ = train_val_test_dict['train']['X_scaled']
        y_train_ = train_val_test_dict['train']['y_scaled']
        X_val_ = train_val_test_dict['val']['X_scaled']
        y_val_ = train_val_test_dict['val']['y_scaled']
        X_test_ = train_val_test_dict['test']['X_scaled']
        y_test_ = train_val_test_dict['test']['y_scaled']

        # scalers

        scaler_train = train_val_test_dict['train']['scaler']
        scaler_val = train_val_test_dict['val']['scaler']
        scaler_test = train_val_test_dict['test']['scaler']

        start_train = time.time()
        model = GRU_(n_units = params[key]['H'], l1_reg=params[key]['l1_reg'], seed=0, x_train_=X_train_, n_steps_ahead=n_steps_ahead)
        model.fit(X_train_, y_train_, epochs=max_epochs, validation_data=(X_val_, y_val_),
                  batch_size=batch_size, callbacks=[es], shuffle=False)
        end_train = time.time()
        params[key]['model'] = model
        params[key]['train_time'] = end_train - start_train


    for key in params.keys():
        model = params[key]['model']

        params[key]['pred_train'] = model.predict(X_train_, verbose=1)
        params[key]['MSE_train'] = mean_squared_error(y_train_, params[key]['pred_train'])

        params[key]['pred_val'] = model.predict(X_val_, verbose=1)
        params[key]['MSE_val'] = mean_squared_error(y_val_, params[key]['pred_val'])
        
        params[key]['pred_test'] = model.predict(X_test_, verbose=1) 
        params[key]['MSE_test'] = mean_squared_error(y_test_, params[key]['pred_test'])

        params[key]['pred_train_scaled'] = scaler_train.inverse_transform(params[key]['pred_train'])
        params[key]['y_train_scaled'] = scaler_train.inverse_transform(y_train_)

        params[key]['pred_val_scaled'] = scaler_val.inverse_transform(params[key]['pred_val'])
        params[key]['y_val_scaled'] = scaler_val.inverse_transform(y_val_)

        params[key]['pred_test_scaled'] = scaler_test.inverse_transform(params[key]['pred_test'])
        params[key]['y_test_scaled'] = scaler_test.inverse_transform(y_test_)


        # Record the MSE, MAE, and R2 metrics in the pickle file
        params[key]['MSE_train_scaled'] = mean_squared_error(params[key]['pred_train_scaled'], params[key]['y_train_scaled'])
        params[key]['MSE_val_scaled'] = mean_squared_error(params[key]['pred_val_scaled'], params[key]['y_val_scaled'])
        params[key]['MSE_test_scaled'] = mean_squared_error(params[key]['pred_test_scaled'], params[key]['y_test_scaled'])

        params[key]['MAE_train_scaled'] = mean_absolute_error(params[key]['pred_train_scaled'], params[key]['y_train_scaled'])
        params[key]['MAE_val_scaled'] = mean_absolute_error(params[key]['pred_val_scaled'],params[key]['y_val_scaled'])
        params[key]['MAE_test_scaled'] = mean_absolute_error(params[key]['pred_test_scaled'],params[key]['y_test_scaled'])

        params[key]['R2_train_scaled'] = r2_score(params[key]['pred_train_scaled'], params[key]['y_train_scaled'])
        params[key]['R2_val_scaled'] = r2_score(params[key]['pred_val_scaled'], params[key]['y_val_scaled'])
        params[key]['R2_test_scaled'] = r2_score(params[key]['pred_test_scaled'], params[key]['y_test_scaled'])
        

        # Record the mean and stadard dev for StandardScaler(), to be used as scaler for user-input values
        for split in ['train', 'val', 'test']:
            for stat in ['mean', 'std']:
                params[key][f'scaler_{split}_{stat}'] = train_val_test_dict[split][f'scaler_{stat}']
        
        params[key]['model'].save(f'app/pages/models/hdf5-{dataset_}-' + key + '.hdf5', overwrite=True)  # creates a HDF5 file
        with open(f'app/pages/models/{dataset_}_models_{model_}.pkl', 'wb') as f:
            pickle.dump(params, f)

def plotAveragedPrediction(predictions, ground_truth):
    
    # Example: Overlapping predictions (4-day forecasts) and ground truth (as continuous values)
    all_predictions = predictions
    ground_truth_4_arrays = ground_truth

    # Convert ground truth into a single continuous array
    ground_truth_continuous = []
    for i in range(len(ground_truth_4_arrays) - 1):
        ground_truth_continuous.append(ground_truth_4_arrays[i, 0])
        ground_truth_continuous.extend(ground_truth_4_arrays[-1])
    
    ground_truth_continuous = np.array(ground_truth_continuous)

    # Time axis for ground truth
    time_axis_truth = np.arange(len(ground_truth_continuous))

    # Initialize arrays for averaged predictions
    time_series_length = len(all_predictions) + 3  # Include overlapping days
    predicted_full = np.zeros(time_series_length)
    overlap_count = np.zeros(time_series_length)  # To average overlapping predictions

    # Fill the arrays for predictions
    for i, forecast in enumerate(all_predictions):
        predicted_full[i:i+4] += forecast
        overlap_count[i:i+4] += 1

    # Average overlapping predictions
    predicted_full /= overlap_count

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot ground truth as a single continuous line
    plt.plot(time_axis_truth, ground_truth_continuous, label="Ground Truth", linestyle="-", marker="o", color="black")

    # Plot the averaged predictions
    time_axis_predictions = np.arange(time_series_length)
    plt.plot(time_axis_predictions, predicted_full, label="Averaged Predictions", linestyle="--", color="blue")

    # Finalize the plot
    plt.title("Averaged Predictions vs Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

def predictModel():
    return None

def load_and_predict(tenor: str, model: str, df: pd.DataFrame):
    
    with open(f"app/pages/models/{tenor.upper()}_models_{model.lower()}.pkl", "rb") as file:
        params_model = pickle.load(file)
    
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = params_model[model.lower()]['scaler_train_mean'], params_model[model.lower()]['scaler_train_std']

    # transform the dataframe
    df_scaled = scaler.transform(df)
    
    # reformat the scaled data frame into X (input), y(target)
    X_scaled, y_scaled = reformat_to_arrays(df_scaled, n_steps = 15, n_steps_ahead = 4)
    
    # predict with the model
    
    model = params_model[model.lower()]['model']
    predictions = model.predict(X_scaled)
    predictions = scaler.inverse_transform(predictions)
    ground_truth = scaler.inverse_transform(y_scaled)
    
    return params_model, predictions, ground_truth

# reads the pickle file for the given model and dataset
# dataset: 2YR or 10YR
# model: RNN, GRU, LSTM
def load_params(dataset_, model_):

    with open(f"app/pages/models/{dataset_.upper()}_models_{model_.lower()}.pkl", "rb") as file:
        params_model = pickle.load(file)

    return params_model