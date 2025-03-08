import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
warnings.simplefilter('ignore')
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import numpy as np
import pandas as pd
from numpy import array
import math
from copy import deepcopy
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
from itertools import islice

import matplotlib.dates as mdates


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import LSTM, Dense ,BatchNormalization,Dropout,SimpleRNN,LeakyReLU,Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation
def custom_activation(x, beta = 0.5):
        return (K.sigmoid(beta * x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

from prophet import Prophet
import torch
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator

def cal_mape(actual_data, fit_data, term):
    #term  = 12
    lastActualData = actual_data[-term:]
    lastFitData    = fit_data[-term:]        
    i = 0
    count = 0
    ape = 0
    for actualVal in lastActualData:
        if(actualVal != 0):
            val = (abs(actualVal - lastFitData[i])/actualVal)*100
            #print(val)
            if(val <100):
                count = count + 1
                ape  = ape + val       
        i = i+1
        
    if(count!=0):
        mape_cal = (ape/count)
    else:
        mape_cal=99
            
    if(math.isnan(mape_cal)):        
            return 0
    else:      
        return mape_cal
    
def evaluate_arima_order (data):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = range(1,4)
    d = range(1,3)
    q = range(1,4)
        
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))
    #print(pdq)
        
    best_aic = np.inf
    best_pdq = None
    temp_model = None
    for param in pdq:   
        temp_model = ARIMA(data,order=param)
        results = temp_model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = param 
    print("Best ARIMA {} model - AIC:{}".format(best_pdq,best_aic))  
      
    return best_pdq

def ARIMA_Model(step,orig_data,best_pdq):      
        
    mod = ARIMA(orig_data, order=best_pdq,enforce_stationarity=False, enforce_invertibility=False)
    model_fit = mod.fit()  
        
    forecasted_value = model_fit.forecast(steps = step)        
    print("forecasted Value",forecasted_value)
    
    return np.round(forecasted_value,3),model_fit

def evaluate_sarima_order (data,time_freq):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))        
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], time_freq) for x in list(itertools.product(p, d, q))]
        
    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    temp_model = None
    for param in pdq:   
        for param_seasonal in seasonal_pdq: 
            temp_model = SARIMAX(data,order=param,seasonal_order = param_seasonal,enforce_invertibility=True,enforce_stationarity=False)
            results = temp_model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
    print("Best ARIMA {} x {} model - AIC:{}".format(best_pdq,best_seasonal_pdq,best_aic)) 
    
    return  best_pdq,best_seasonal_pdq
def SARIMA_Model(step,orig_data,best_pdq,best_seasonal_pdq):        
        
    mod = SARIMAX(orig_data,order=best_pdq,seasonal_order=best_seasonal_pdq)
    model_fit = mod.fit(disp=False)     
        
    forecasted_value = model_fit.forecast(steps = step)        
    print("forecasted Value",forecasted_value)

    return np.round(forecasted_value,3),model_fit

def Holt_Winters_Model(step,orig_data,model_type,data_period):  
    #print("In holtz winter")      
    if(model_type=='add'):
        mod = ExponentialSmoothing(orig_data , initialization_method= 'legacy-heuristic',seasonal_periods=data_period,trend='add', seasonal='add')
    else:
        mod = ExponentialSmoothing(orig_data ,initialization_method= 'legacy-heuristic',seasonal_periods=data_period ,trend='mul',seasonal='mul')
        
    model_fit = mod.fit(optimized=True,use_brute=True)  
    #print(model_fit.summary())
    
    forecasted_value = model_fit.forecast(steps = step)
    #print("forecasted Value",forecasted_value)       
 
    return np.round(forecasted_value,3),model_fit
# ********** Random Forest Regression Model (RFR Model)
def RFR_Model(step,end_date_obj,train_data2,sales_column,date_column,value_columns):
    train_data =train_data2.copy()
    train_data = train_data.reset_index()
    train_data['year'] = train_data[date_column].dt.year
    train_data['month'] = train_data[date_column].dt.month
    #train_data['day'] = train_data[date_column].dt.day   
    if(len(value_columns)>0):
        sales_column = [sales_column]+value_columns
    X_train = train_data[['year', 'month']]
    orig_data = train_data[sales_column] 
    
    model = RandomForestRegressor(random_state= 42,n_jobs=-1)

    model_fit = model.fit(X_train,orig_data)
    fitted_data = model_fit.predict(X_train)
    
    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[date_column])
        
    forecast_date['year'] = forecast_date[date_column].apply(lambda x: int(x[-4:]))
    forecast_date['month'] = forecast_date[date_column].apply(lambda x: int(x[3:5]))
    X_test=forecast_date[['year', 'month']]    
    forecasted_value = model_fit.predict(X_test)
    temp=[]
    fitt_temp=[]
    if(len(value_columns)>0):
        for x in forecasted_value:
            temp.append(x[0])
        forecasted_value=temp
        for x in fitted_data:            
            fitt_temp.append(x[0])
            fitted_data=fitt_temp
    print(" RFR_Model forecasted Value",forecasted_value)       
    return np.round(forecasted_value,3),model_fit,fitted_data
    
# ********** Random Forest Regression Model (RFR Model)
def SVR_Model(step,end_date_obj,train_data2,sales_column,date_column):
    train_data =train_data2.copy()
    train_data = train_data.reset_index()
    train_data['year'] = train_data[date_column].dt.year
    train_data['month'] = train_data[date_column].dt.month
    #train_data['day'] = train_data[date_column].dt.day   
    
    X_train = train_data[['year', 'month']]
    orig_data = train_data[sales_column] 
    
    # Define the parameter grid for hyper-parameter tuning
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100, 200,300,400,500,600,700,800,900,1000],
        'gamma': [0.01, 0.1, 0.5, 1, 'scale', 'auto'],
        'epsilon': [0.001, 0.005, 0.01, 0.1, 0.5]
    }
    # Initialize SVR model
    svr = SVR()
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    # Fit the grid search to the data
    grid_search.fit(X_train, orig_data)
    model = grid_search.best_estimator_

    model_fit = model.fit(X_train,orig_data)
    fitted_data = model_fit.predict(X_train)
    
    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[date_column])
        
    forecast_date['year'] = forecast_date[date_column].apply(lambda x: int(x[-4:]))
    forecast_date['month'] = forecast_date[date_column].apply(lambda x: int(x[3:5]))
    X_test=forecast_date[['year', 'month']]    
    forecasted_value = model_fit.predict(X_test)
    
    return np.round(forecasted_value,3),model_fit,fitted_data
    
# ********** Random Forest Regression Model (RFR Model)
def XGBR_Model(step,end_date_obj,train_data2,sales_column,date_column,value_columns):
    train_data =train_data2.copy()
    train_data = train_data.reset_index()
    train_data['year'] = train_data[date_column].dt.year
    train_data['month'] = train_data[date_column].dt.month
    #train_data['day'] = train_data[date_column].dt.day   
    if(len(value_columns)>0):
        sales_column = [sales_column]+value_columns
    X_train = train_data[['year', 'month']]
    orig_data = train_data[sales_column] 
    
    #max_est=int(max(orig_data))
    #model = XGBRegressor(booster='gbtree', objective='reg:squarederror', max_depth=2,n_estimators=(max_est+int(max_est*0.10)), random_state=3, n_jobs=-1) 
    
    model = XGBRegressor(booster='gbtree', objective='reg:squarederror', max_depth=3,random_state= 42,n_jobs=-1)

    model_fit = model.fit(X_train,orig_data)
    fitted_data = model_fit.predict(X_train)
    
    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[date_column])
        
    forecast_date['year'] = forecast_date[date_column].apply(lambda x: int(x[-4:]))
    forecast_date['month'] = forecast_date[date_column].apply(lambda x: int(x[3:5]))
    X_test=forecast_date[['year', 'month']]    
    forecasted_value = model_fit.predict(X_test)
    temp=[]
    fitt_temp=[]
    if(len(value_columns)>0):
        for x in forecasted_value:
            temp.append(x[0])
        forecasted_value=temp
        for x in fitted_data:            
            fitt_temp.append(x[0])
            fitted_data=fitt_temp
    return np.round(forecasted_value,3),model_fit,fitted_data
# ********** Voting Regression Model (Voting Model)
def Voting_Model(step,end_date_obj,train_data2,sales_column,date_column,value_columns):
    from sklearn.multioutput import MultiOutputRegressor

    train_data =train_data2.copy()
    train_data = train_data.reset_index()
    train_data['year'] = train_data[date_column].dt.year
    train_data['month'] = train_data[date_column].dt.month
    #train_data['day'] = train_data[date_column].dt.day   
    sales_column2=deepcopy(sales_column)
    if(len(value_columns)>0):
        sales_column = [sales_column]+value_columns
    X_train = train_data[['year', 'month']]
    orig_data = train_data[sales_column] 
    orig_data2 = train_data[sales_column2] 
    
    max_est=int(max(orig_data2))  
        
    reg1 = GradientBoostingRegressor(n_estimators=(max_est+int(max_est*0.10)), learning_rate=0.1, max_depth=2, random_state=2,loss='squared_error')
    reg2 = RandomForestRegressor(random_state= 24,n_jobs=-1)
    reg3 = XGBRegressor(booster='gbtree', objective='reg:squarederror', max_depth=2,n_estimators=(max_est+int(max_est*0.10)), random_state=3, n_jobs=-1)        
    model = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2),('xgb', reg3)])
    
    if(len(value_columns)>0):
        model = MultiOutputRegressor(model)
    
    model_fit = model.fit(X_train,orig_data)
    fitted_data = model_fit.predict(X_train)
    # Predict multiple outputs

    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[date_column])
        
    forecast_date['year'] = forecast_date[date_column].apply(lambda x: int(x[-4:]))
    forecast_date['month'] = forecast_date[date_column].apply(lambda x: int(x[3:5]))
    X_test=forecast_date[['year', 'month']]    
    forecasted_value = model_fit.predict(X_test)
    temp=[]
    fitt_temp=[]
    if(len(value_columns)>0):
        for x in forecasted_value:
            temp.append(x[0])
        forecasted_value=temp
        for x in fitted_data:            
            fitt_temp.append(x[0])
            fitted_data=fitt_temp
    return np.round(forecasted_value,3),model_fit,fitted_data  
# ********** Random Forest Regression Model for LSTM/CNN/RNN
def RFR_DL_Model(step,end_date_obj,train_data2,sales_column,date_column):
    print("IN RFR_DL_Model ",sales_column)
    train_data =train_data2.copy()
    train_data = train_data.reset_index()
    train_data['year'] = train_data[date_column].dt.year
    train_data['month'] = train_data[date_column].dt.month
    #train_data['day'] = train_data[date_column].dt.day   
    
    X_train = train_data[['year', 'month']]
    orig_data = train_data[sales_column] 
    
    model = RandomForestRegressor(random_state= 42,n_jobs=-1)

    model_fit = model.fit(X_train,orig_data)
    fitted_data = model_fit.predict(X_train)
    
    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[date_column])
        
    forecast_date['year'] = forecast_date[date_column].apply(lambda x: int(x[-4:]))
    forecast_date['month'] = forecast_date[date_column].apply(lambda x: int(x[3:5]))
    X_test=forecast_date[['year', 'month']]    
    forecasted_value = model_fit.predict(X_test)
    
    #print(" RFR_Model forecasted Value",forecasted_value)       
    return np.round(forecasted_value,3),model_fit,fitted_data
def RNN_Model(step,end_date_obj,train_data2,sales_column,date_column):
    df2 =train_data2.copy()
    df2 = df2.reset_index()
    #print("In LSTM_MOdel")
    data=df2[[date_column, sales_column]]
    # Reshape data for LSTM
    train_data = data[sales_column].values.reshape(-1, 1)
    test_data = data[sales_column].values.reshape(-1, 1)
    #print("Shape : ",train_data.shape)   
   
    model = Sequential()
    model.add(SimpleRNN(units=64, activation='relu',return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(Dropout(0.2)) 
    model.add(SimpleRNN(units=32,activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model_fit = model.fit(train_data, train_data, epochs=500, batch_size=64,verbose=0)
    
    # Make predictions        
    forecast_lstm = model.predict(test_data)
    forecast_lstm = forecast_lstm.reshape(-1,1)
    
    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[sales_column])
    
    temp_fvalue,_,_=RFR_DL_Model(step,end_date_obj,train_data2,sales_column,date_column)
    #print(">>>>>>>>>>>>>>>>>>>>>> ",temp_fvalue)
    
    input_x = temp_fvalue.reshape(-1, 1)
    #print("input_x",input_x)
    forecasted_value = model.predict(input_x, verbose=0)
    forecasted_value = forecasted_value.reshape(-1,1)
    #print("forecasted_value",forecasted_value)      

    return np.round(forecasted_value.reshape(-1),3),model_fit,forecast_lstm.reshape(-1)    
 
def CNN_Model(step,end_date_obj,train_data2,sales_column,date_column):
    df2 = train_data2.copy()
    df2 = df2.reset_index()
    
    data = df2[[date_column, sales_column]]
    train_data = data[sales_column].values.reshape(-1, 1)
    test_data = data[sales_column].values.reshape(-1, 1)
    
    # Reshape data for CNN (samples, time steps, features)
    train_data = train_data.reshape(train_data.shape[0], 1, 1)
    test_data = test_data.reshape(test_data.shape[0], 1, 1)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model_fit = model.fit(train_data, train_data, epochs=500, batch_size=64, verbose=0)
    
    # Make predictions
    forecast_cnn = model.predict(test_data)
    forecast_cnn = forecast_cnn.reshape(-1, 1)
    
    fdateList = []
    for i in range(1, step + 1):
        date_after_month = end_date_obj + relativedelta(months=i)
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))
    
    forecast_date = pd.DataFrame(fdateList, columns=[sales_column])
    
    temp_fvalue, _, _ = RFR_DL_Model(step, end_date_obj, train_data2, sales_column, date_column)
    
    input_x = temp_fvalue.reshape(-1, 1, 1)
    forecasted_value = model.predict(input_x, verbose=0)
    forecasted_value = forecasted_value.reshape(-1, 1)
    
    return np.round(forecasted_value.reshape(-1), 3), model_fit, forecast_cnn.reshape(-1)   
    
def LSTM_Model(step,end_date_obj,train_data2,sales_column,date_column,value_columns):
    df2 =train_data2.copy()
    df2 = df2.reset_index()
    #print("In LSTM_MOdel")
    ln=1
    sales_column2=deepcopy(sales_column)
    if(len(value_columns)>0):
        sales_column = [sales_column]+value_columns
        ln=len(sales_column)
        data=df2[[date_column]+sales_column]
    else:
        data=df2[[date_column, sales_column]]
    # Reshape data for LSTM
    train_data = data[sales_column].values.reshape(-1, ln)
    test_data = data[sales_column].values.reshape(-1, ln)
    #print("Shape : ",train_data.shape)
    
    #leaky_relu_layer = tf.keras.layers.LeakyReLU(alpha=0.1) 
    model = Sequential()    
    model.add(LSTM(units=128, activation=LeakyReLU(alpha=0.1),input_shape=(train_data2.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units=64,activation='relu', return_sequences=True))
    model.add(Dropout(0.2))    
    model.add(LSTM(units=32,activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    # Create and train LSTM model
    # model = Sequential()
    # model.add(LSTM(units=64, activation=LeakyReLU(alpha=0.1), return_sequences=True, input_shape=(train_data.shape[1], 1)))
    # model.add(LSTM(units=32,activation='relu'))
    # model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10) # Stop training when validation loss stops decreasing for 10 epochs
    model_fit = model.fit(train_data, train_data, validation_data=(train_data, train_data), epochs=2000, batch_size=32,shuffle=False, verbose=0,callbacks=[early_stopping])
    
    #model.fit(train_data, train_data, epochs=100, batch_size=36,verbose=0)
    
    # Make predictions        
    forecast_lstm = model.predict(test_data)
    forecast_lstm = forecast_lstm.reshape(-1,1)
    
    #print(forecast_lstm)
    # Evaluate performance
    #print("step",step)
    fdateList=[]
    for i in range(1,step+1):
        date_after_month = end_date_obj+ relativedelta(months=i) 
        fdateList.append(date_after_month.strftime("%d-%m-%Y"))            
    forecast_date=pd.DataFrame(fdateList,columns=[sales_column2])
    
    temp_fvalue,_,_=RFR_DL_Model(step,end_date_obj,train_data2,sales_column2,date_column)
    #print(">>>>>>>>>>>>>>>>>>>>>> ",temp_fvalue)
    
    input_x = temp_fvalue.reshape(-1, 1)
    #print("input_x",input_x)
    forecasted_value = model.predict(input_x, verbose=0)
    forecasted_value = forecasted_value.reshape(-1,1)
    #print("forecasted_value",forecasted_value)      
    

    #mape = round(igsa_mape(data[sales_column],forecast_lstm.reshape(-1),len(data[sales_column])),5) 
    #print('IGSA LSTM MAPE:', mape)
    
    return np.round(forecasted_value.reshape(-1),3),model_fit,forecast_lstm.reshape(-1)    

# ********** Prophet Model ******************************                   
def PROPHET_Model(step,end_date_obj,train_data2,sales_column,date_column,time_freq):  
    #print("In PROPHET_MOdel")
    df2 =train_data2.copy()
    df2 = df2.reset_index()
    
    orig_data=df2[[date_column,sales_column]]
    orig_data.columns = ['ds', 'y']
    
    #print("time_freq : ",time_freq)
    model = Prophet(seasonality_mode='additive',yearly_seasonality=12).fit(orig_data)
    
    
    future = model.make_future_dataframe(periods = step,freq=time_freq)
    forecast = model.predict(future)
    fitted_data = forecast['yhat'][:-step].values
        
    forecasted_value = forecast['yhat'][-step:].values    
    #print("PROPHET forecasted Value",np.round(forecasted_value,3))
    
    return np.round(forecasted_value,3),future,fitted_data

def _get_lag_llama_predictions(dataset, prediction_length, num_samples=100):
    # Load the checkpoint with map_location set to 'cpu'
    ckpt = torch.load("lag-llama.ckpt", map_location=torch.device('cpu'))
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    # Initialize the estimator with the required arguments
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32,
        
        # scaling="mean",
        #nonnegative_pred_samples=True,
        aug_prob=0,
        lr=0.0005,
        
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        
        # linear positional encoding scaling
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (32 + prediction_length) / estimator_args["context_length"]),
        },
        
        batch_size=32,
        num_parallel_samples=num_samples,
        trainer_kwargs = {"max_epochs": 200,}, # <- lightning trainer arguments
    )


    # Create the lightning module, transformation, and predictor
    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    # Generate forecasts and time series predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss,predictor

def _get_lag_llama_dataset(data_column,dataset):
    # avoid mutations
    dataset = dataset.copy()
    
    # convert numerical columns to `float32`
    for col in dataset.columns:      
        if dataset[col].dtype != "object" and not pd.api.types.is_string_dtype(
            dataset[col]
        ):
            dataset[col] = dataset[col].astype("float32")

    # create a `PandasDataset`
    backtest_dataset = PandasDataset(dict(dataset))
    #print(backtest_dataset)
    return backtest_dataset

def Lagllama_Model(step,end_date_obj,train_data2,sales_column,date_column,time_freq):
    df =train_data2.copy()
    df=df[[sales_column]]
    #print(len(df))
    backtest_dataset = _get_lag_llama_dataset(sales_column,dataset=df)
    prediction_length = step  # prediction length
    num_samples = len(df) # 36 is sampled from the distribution for each timestep
    forecasts, tss,model_fit = _get_lag_llama_predictions(
    backtest_dataset, prediction_length, num_samples
    )
    # Extract forecasted mean values
    forecast_index = pd.date_range(start=df.index[-1], periods=prediction_length+1, freq=time_freq)[1:]

    
    forecast_values = [forecast.mean for forecast in forecasts]
    forecast_df = pd.DataFrame(forecast_values[0], index=forecast_index, columns=['Forecast'])
    
    forecasted_value=forecast_df['Forecast'].values
    print("Lag-llama forecasted Value",np.round(forecasted_value,3))
    
    # fitted_values=[]
    # for forecast in forecasts[0].samples:
    #     fitted_values.append((forecast[1]+forecast[2])/2)
    
    fitted_values=tss[0].values.reshape(1,-1)[0]
    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
    mae = agg_metrics['abs_error']
    mape = agg_metrics['MAPE']*100
    rmse = agg_metrics['RMSE']
    if(mape>99):
        mape=99
    print("Lag LLama : ",mae,mape,rmse)

    return np.round(forecasted_value,3),model_fit,fitted_values,mae,mape,rmse
def _get_lag_llama_predictions_tuned(dataset, prediction_length, num_samples=100):
    # Load the checkpoint with map_location set to 'cpu'
    ckpt = torch.load("lag-llama.ckpt", map_location=torch.device('cpu'))
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    # Initialize the estimator with the required arguments
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32,
        
        # scaling="mean",
        #nonnegative_pred_samples=True,
        aug_prob=0,
        lr=0.0005,
        
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        
        # linear positional encoding scaling
        # rope_scaling={
        #     "type": "linear",
        #     "factor": max(1.0, (32 + prediction_length) / estimator_args["context_length"]),
        # },
        
        batch_size=32,
        num_parallel_samples=num_samples,
        trainer_kwargs = {"max_epochs": 50,}, # <- lightning trainer arguments
    )


    # Create the lightning module, transformation, and predictor
    #lightning_module = estimator.create_lightning_module()
    #transformation = estimator.create_transformation()
    #predictor = estimator.create_predictor(transformation, lightning_module)
    predictor = estimator.train(dataset, cache_data=True, shuffle_buffer_length=1000)
    
    # Generate forecasts and time series predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss,predictor
def Lagllama_Model_Tuned(step,end_date_obj,train_data2,sales_column,date_column,time_freq):
    df =train_data2.copy()
    df=df[[sales_column]]
    #print(len(df))
    backtest_dataset = _get_lag_llama_dataset(sales_column,dataset=df)
    prediction_length = step  # prediction length
    num_samples = len(df) # 36 is sampled from the distribution for each timestep
    forecasts, tss,model_fit = _get_lag_llama_predictions_tuned(
    backtest_dataset, prediction_length, num_samples
    )
    # Extract forecasted mean values
    forecast_index = pd.date_range(start=df.index[-1], periods=prediction_length+1, freq=time_freq)[1:]

    
    forecast_values = [forecast.mean for forecast in forecasts]
    forecast_df = pd.DataFrame(forecast_values[0], index=forecast_index, columns=['Forecast'])
    
    forecasted_value=forecast_df['Forecast'].values
    print("Lag-llama forecasted Value",np.round(forecasted_value,3))
    
    # fitted_values=[]
    # for forecast in forecasts[0].samples:
    #     fitted_values.append((forecast[1]+forecast[2])/2)
    
    fitted_values=tss[0].values.reshape(1,-1)[0]
    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
    mae = agg_metrics['abs_error']
    mape = agg_metrics['MAPE']*100
    rmse = agg_metrics['RMSE']
    if(mape>99):
        mape=99
    print("Lag LLama : ",mae,mape,rmse)

    return np.round(forecasted_value,3),model_fit,fitted_values,mae,mape,rmse