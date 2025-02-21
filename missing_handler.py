'''
Created on 20-January-2025

@author: AFTAB HASSAN
'''

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import KNNImputer
    
def mean_case(df):   
    print("I am in mean")     
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    #non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns            
    # Mean imputation for numeric columns
    df_filled = df.copy()
    #print(df)
    mean_imputer = SimpleImputer(strategy='mean')
    df_filled[numeric_cols] = mean_imputer.fit_transform(df[numeric_cols])          
    return df_filled
    
def median_case(df):    
    print("I am in median")     
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    #non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns            
    # Mean imputation for numeric columns
    mean_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = mean_imputer.fit_transform(df[numeric_cols])           
    return df
    
def mode_case(df):    
    print("I am in mode")     
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    #non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns            
    # Mean imputation for numeric columns
    mean_imputer = SimpleImputer(strategy='most_frequent')
    df[numeric_cols] = mean_imputer.fit_transform(df[numeric_cols])            
    return df
    
def forward_fill_case(df):    
    print("I am in Forward Fill")        
    df= df.fillna(method='ffill').fillna(method='bfill')
    return df
    
def backward_fill_case(df):    
    print("I am in Backward Fill")     
    df= df.fillna(method='bfill').fillna(method='ffill')               
    return df  

def linear_interpolation_case(df):    
    print("I am in linear interpolation") 
    column_names = df.select_dtypes(include=['float64', 'int64']).columns
    df_filled = df.copy()
    for column in column_names:                   
        df_filled[column]=df[column].interpolate(method='linear')
        df_filled[column]=df_filled[column].fillna(df_filled[column].mean())
    
    
    return df_filled
    
def polynomial_interpolation_case(df):    
    print("I am in Polynomial Interpolation")  
    degree=3
    column_names = df.select_dtypes(include=['float64', 'int64']).columns
    df_filled = df.copy()
    
    for column in column_names:
        # Get the indices where values are not missing
        known_idx = df[column].dropna().index
        known_values = df[column].dropna().values
        
        # Get the indices of missing values
        missing_idx = df[column][df[column].isnull()].index
        
        if len(known_idx) >= degree + 1:
            # Perform polynomial interpolation using known values
            poly_interp = np.polyfit(known_idx, known_values, degree)
            poly_func = np.poly1d(poly_interp)
            
            # Predict the missing values
            df_filled.loc[missing_idx, column] = poly_func(missing_idx)
        else:
            print(f"Not enough points to perform polynomial interpolation for {column}") 
            df_filled=linear_interpolation_case(df)               
    
    return df_filled
        
def k_nearest_case(df):
    print("I am in K-Nearest")  
    if(len(df)>6):
        n_neighbors=3
    else:
        n_neighbors=1
        
    df_filled = df.copy()
    column_names = df.select_dtypes(include=['float64', 'int64']).columns
    for column in column_names:  
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_temp = pd.DataFrame(imputer.fit_transform(df[[column]]), columns=[column])
        df_filled[[column]]=df_temp.values.tolist()        
    return df_filled 
    
    
       
def choose_best_method(df):
    column_names = df.select_dtypes(include=['float64', 'int64']).columns
    for column in column_names:
        
        if not df[column].isnull().all():
            if not df[column].mean():
                return mean_case(df)
        if not df[column].isnull().all():
            if not df[column].median():
                return median_case(df)
        if not df[column].mode().empty:
            return mode_case(df)
        # K-Nearest Neighbor imputation
        try:
            imputer = KNNImputer(n_neighbors=3)
            df_imputed = pd.DataFrame(imputer.fit_transform(df[[column]]), columns=[column])
            if not df_imputed.isnull().any().any():
                return k_nearest_case(df)
        except Exception:
            continue
        
        if not df[column].fillna(method='ffill').isnull().any():
            return forward_fill_case(df)
        if not df[column].fillna(method='bfill').isnull().any():
            return backward_fill_case(df)
        # Linear Interpolation
        try:
            df_interp = df[column].interpolate(method='linear')
            if not df_interp.isnull().any():
                return linear_interpolation_case(df)
        except Exception:
            continue
        try:
            df_poly_interp = df[column].interpolate(method='polynomial', order=3)
            if not df_poly_interp.isnull().any():
                return polynomial_interpolation_case(df)
        except Exception:
            continue

    # If no suitable method is found, return 'Default'
    return mean_case(df)
        
        


