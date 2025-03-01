import streamlit as st
import pandas as pd
import requests
import json
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error, r2_score
import numpy as np

# Azure Function API Endpoint
API_URL = "https://wilp-05329-app.azurewebsites.net/api/wilp_models?code=fxFOtmWcM1ZRV5hvAJHW9XWl5jneuOvb5Bg9Irm8tPYCAzFuzqsxdw=="

# Streamlit App
st.title("Azure Function API Tester")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #st.write("Uploaded Data:")
    #st.dataframe(df)
    
    # Check if the required columns exist
    if "ALL_DATE" in df.columns and "QTY" in df.columns:
        # Convert dataframe to the required JSON format
        data_payload = {"data": df[["ALL_DATE"]].to_dict(orient="records")}
        
        # Send request to Azure Function API
        response = requests.post(API_URL, json=data_payload)
        
        # Display response
        if response.status_code == 200:
            st.success("API Response:")
            response_data = response.json()
            #st.write("Raw API Response:", response_data)  # Debugging step
            
            # Convert response to DataFrame
            response_df = pd.DataFrame(response_data)
            response_df.index = df["ALL_DATE"]
            
            # Identify correct forecast column
            forecast_column = None
            for col in response_df.columns:
                if "predictions" in col.lower() :
                    forecast_column = col
                    break
            
            if forecast_column:
                response_df["QTY"] = df["QTY"].values
                
                # Calculate error metrics
                mae = round(mean_absolute_error(response_df["QTY"], response_df[forecast_column]),2)
                mape = round(mean_absolute_percentage_error(response_df["QTY"], response_df[forecast_column])* 100,2)
                rmse = round(np.sqrt(mean_squared_error(response_df["QTY"], response_df[forecast_column])),2)
                r2 = round(r2_score(response_df["QTY"], response_df[forecast_column]),2)
                
                # Display results
                st.dataframe(response_df)
                st.write(f"**Mean Absolute Error (MAE):** {mae}")
                st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape}%")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")
                st.write(f"**R² Score:** {r2}")
            else:
                st.error("Forecasted column not found in API response. Check API response format.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    else:
        st.error("CSV file must contain 'ALL_DATE' and 'Qty' columns.")
