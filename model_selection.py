'''
Created on 04-February-2025

@author: AFTAB HASSAN
'''

import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
#from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

# Suppress warnings globally for this script
#Sos.environ["PYTHONWARNINGS"] = "ignore"

threshold_autocorr = 0.3  # Threshold for long-term dependencies
threshold_variability = 0.1  # Threshold for noise/variability
        
## criteria flag to decide model  
DATAVARIATE=False   # False for Univariate and True for Multivariate Data
STATIONARITY=False  # False means data has not Stationarity and True means data has Stationarity
SEASONALITY=False   # False means data has not Seasonality and True means data has Seasonality
TRENDS=False        # False means data has not Trends and True means data has Trends
LEVEL=False         # False means data has not Level and True means data has Level
COMPLEXITY=False    # False means data has not Complexity and True means data has Complexity
DEPENDENCITY=False  # False means data has not Long-term Dependencie and True means data has Long-term Dependencie
    
def check_datavariate(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    #print(df)
    data_column_numbers=len(numeric_cols)
    if data_column_numbers > 1:
        return False  # Multivariate
    else:
        return True   # Univariate    
        
def check_stationarity(timeseries):
    #print(">>>>>>>>>>>>>>> ",timeseries)
    if np.max(timeseries) == np.min(timeseries):
        return True  # Stationary since all values are constant
    else:
        try:
            # Perform Augmented Dickey-Fuller test to check for stationarity
            result = adfuller(timeseries)
            p_value = result[1]  # Extract the p-value from the ADF test result
            
            # A p-value > 0.05 indicates non-stationarity (null hypothesis cannot be rejected)
            if p_value > 0.05:
                return False  # Not stationary
            else:
                return True   # Stationary
        except Exception as e:
            #print(e)
            return False  # Not stationary
    
def check_seasonality(df,numeric_cols,freq):  
    if(len(df)>=freq*2):  
        decomposition = seasonal_decompose(df[numeric_cols], period=freq, model='additive')
        
        seasonal = decomposition.seasonal
        if np.std(seasonal) > 0.01:
            return True  # Seasonal
        else:
            return False  # Not seasonal
    else:
        return False  # Not seasonal
            
def check_trends(df,numeric_cols,freq):     
    if(len(df)>=freq*2):  
        decomposition = seasonal_decompose(df[numeric_cols], period=freq, model='additive')
        
        trend = decomposition.trend
        trend = trend.fillna(trend.mean())
        
        if np.std(trend) > 0.01:
            return True  # Seasonal
        else:
            return False  # Not seasonal
    else:
        return False  # Not seasonal
        
# Function to check for levels in the time series data
def check_levels(df, window=12, threshold=0.05):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    levels_detected = {}        
    for col in numeric_cols:
        # Calculate the rolling mean over the specified window
        rolling_mean = df[col].rolling(window=window).mean()            
        # Calculate the absolute difference between the rolling mean and the overall mean
        overall_mean = df[col].mean()
        diff = np.abs(rolling_mean - overall_mean)            
        # Detect if the mean changes exceed a specified threshold
        if np.mean(diff) > threshold * overall_mean:
            levels_detected=True  # Level shift detected
        else:
            levels_detected=False  # No significant level shift
            
    return levels_detected
    

def check_long_term_dependencies(df,numeric_cols,threshold=0.4):
    timeseries = df[numeric_cols]
    
    nlag=int(len(timeseries)*0.5)
    # Calculate the ACF values
    acf_values = acf(timeseries, nlags=nlag)
        
    # Check if ACF values exceed the threshold beyond lag 1 (ignore lag 0)
    if np.any(np.abs(acf_values[1:]) > threshold):
        #print("Long-term dependencies detected.")
        return True
    else:
        #print("No long-term dependencies detected.")
        return False
    
    
def check_autocorrelation(timeseries):
    """
    Check for long-term dependencies using autocorrelation function (ACF).
    """    
    try:
        nlag=int(len(timeseries)*0.5)
        autocorr_values = acf(timeseries, nlags=nlag)
        if np.any(np.abs(autocorr_values[1:]) > threshold_autocorr):
            return True  # Long-term dependencies detected
    except Exception as e:
        print(f"Error: {e}")
        
    return False  # No significant long-term dependencies

def check_non_linear(timeseries):
    """
    Check for non-linear behavior using the Ljung-Box test for autocorrelation.
    """
    try:
        lag=len(timeseries)*0.5
        result = acorr_ljungbox(timeseries, lags=[lag], return_df=True)
        p_value = result['lb_pvalue'].iloc[0]
        if p_value < 0.05:
            return True  # Non-linear patterns detected
    except Exception as e:
        print(f"Error: {e}")
        
    return False  # Linear behavior

def check_variability(timeseries):
    """
    Check for high variability in the time series based on standard deviation.
    """
    
    std_dev = np.std(timeseries.all())
    if std_dev > threshold_variability:
        return True  # High variability (complex)
    return False  # Low variability (less complex)

def check_complexity(df,numeric_cols):
    
    # If the dataset has more than one column, it's multivariate
    if check_datavariate(df):
        #print("The dataset is multivariate, and thus complex.")
        return True

    # Assuming it's univariate if we reach here
    timeseries = df[numeric_cols]  # Select the first column if it's univariate

    # Check for autocorrelation (long-term dependencies)
    if check_autocorrelation(timeseries):
        #print("Long-term dependencies detected. The time series is complex.")
        return True

    # Check for non-linear behavior using the Ljung-Box test
    if check_non_linear(timeseries):
        #print("Non-linear patterns detected. The time series is complex.")
        return True

    # Check for high variability or noise
    if check_variability(timeseries):
        #print("High variability detected. The time series is complex.")
        return True

    # If none of the complexity criteria are met, it's a simpler time series
    #print("The time series is not complex.")
    return False
  
def forecast_model_selection_algorithm(df,freq=12):
    #print("I am in model selection")
    #print(df)
    SelModel={}
    dl_moldels=[]
    stat_models=["ARIMA"]
    ml_models=[]

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    #orig_data = np.asarray(df[numeric_cols]) 
    if(len(numeric_cols)>0):
        numeric_cols=numeric_cols[0]

    # Calling function to check data variate
    DATAVARIATE=check_datavariate(df)
    print("Data Uni-variate flag is ",DATAVARIATE)
    
    # Calling function to check stationarity
    STATIONARITY=check_stationarity(df[numeric_cols])
    print("Stationarity flag is ",STATIONARITY)
    
    # Calling function to check Seasonality
    SEASONALITY=check_seasonality(df,numeric_cols,freq)
    print("Seasonality flag is ",SEASONALITY)
    
    # Calling function to check Trends
    TRENDS=check_trends(df,numeric_cols,freq)
    print("Trends flag is ",TRENDS)
    
    # # Calling function to check Levels
    LEVEL=check_levels(df)
    #print("Levels flag is ",LEVEL)
    
    # Calling function to check Long-Term Dependencies
    DEPENDENCITY=check_long_term_dependencies(df,numeric_cols)
    print("Long-Term Dependencies flag is ",DEPENDENCITY)
    
    # Calling function to check Levels
    COMPLEXITY=check_complexity(df,numeric_cols)
    print("Data COMPLEXITY flag is ",COMPLEXITY)
    
    
    if(DATAVARIATE==True): # Data is Univariate
        if(COMPLEXITY==True):
            if(DEPENDENCITY==True and SEASONALITY):
                dl_moldels.append("LSTM")
                ml_models.append("XGboost Regression")
                stat_models.append("Holt-Winters")
            elif(DEPENDENCITY==True):
                dl_moldels.append("LSTM")
                ml_models.append("XGboost Regression")
        else:       
            if(STATIONARITY==True):
                if(SEASONALITY==True):
                    if(TRENDS==True):
                        stat_models.append("Holt-Winters")
                        ml_models.append("Voting Regression")
                        dl_moldels.append("PROPHET")
                        stat_models.append("SARIMA")
                    else:
                        stat_models.append("SARIMA")
                        stat_models.append("ARIMA")
                        ml_models.append("Random Forest")
                        ml_models.append("XGboost Regression")
                else:
                    stat_models.append("ARIMA")
                    ml_models.append("Random Forest")
                    ml_models.append("XGboost Regression")
            else:
                stat_models.append("ARIMA")
                ml_models.append("Random Forest")
            
    else: # Data is Multivariate
        if(COMPLEXITY==True):
            if(DEPENDENCITY==True):
                dl_moldels.append("LSTM")        
                
                ml_models.append("XGboost Regression")
                ml_models.append("Voting Regression")
            else:
                dl_moldels.append("RNN")  
                dl_moldels.append("CNN")                
                ml_models.append("Random Forest")
                ml_models.append("XGboost Regression")
        else:
            ml_models.append("Random Forest")
            
            
    SelModel['STATS']=list(set(stat_models))
    SelModel['ML']=list(set(ml_models))
    SelModel['DL']=list(set(dl_moldels))
    
    return SelModel