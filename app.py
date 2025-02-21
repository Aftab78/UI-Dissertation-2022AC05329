import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from models import cal_mape,evaluate_arima_order,evaluate_sarima_order
from models import ARIMA_Model,SARIMA_Model,Holt_Winters_Model,RFR_Model,XGBR_Model,Voting_Model,SVR_Model
from models import CNN_Model,RNN_Model,LSTM_Model,PROPHET_Model,Lagllama_Model,Lagllama_Model_Tuned
from missing_handler import mean_case,median_case,mode_case,forward_fill_case,backward_fill_case
from missing_handler import linear_interpolation_case,k_nearest_case,choose_best_method
from model_selection import forecast_model_selection_algorithm
from holtwinter import additive
from st_mui_table import st_mui_table
import io
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from database import save_result_db
if "models" not in st.session_state:
    st.session_state['models'] = None
if "fcr" not in st.session_state:
    st.session_state['fcr'] = None
if "fdr" not in st.session_state:
    st.session_state['fdr'] = None
if "ms" not in st.session_state:
    st.session_state['ms'] = None
if "lmae" not in st.session_state:
    st.session_state['lmae'] = None
if "lmape" not in st.session_state:
    st.session_state['lmape'] = None
if "lrmse" not in st.session_state:
    st.session_state['lrmse'] = None
if "tlmae" not in st.session_state:
    st.session_state['tlmae'] = None
if "tlmape" not in st.session_state:
    st.session_state['tlmape'] = None
if "tlrmse" not in st.session_state:
    st.session_state['tlrmse'] = None    
    
if "forecast_periods" not in st.session_state:
    st.session_state['forecast_periods'] = None
    
# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'missing_handled_data' not in st.session_state:  
    st.session_state['missing_handled_data']=None
if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] = None
if 'decomposition_graph' not in st.session_state:
    st.session_state['decomposition_graph'] = None
if 'forecast_results' not in st.session_state:
    st.session_state['forecast_results'] = None
if 'time_freq' not in st.session_state:
    st.session_state['time_freq'] = None
if 'suggested_models' not in st.session_state:
    st.session_state['suggested_models']=None

    
# Page configuration
st.set_page_config(page_title="Time Series Forecasting System", page_icon="images/icon.png",layout="wide")
# Custom CSS to remove top space
# linear-gradient(to right, #FF5733, #FFC300, #33FF57, #33C1FF, #FF33C1);
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.2rem !important;
        }
        .custom-title {
            font-size: 30px;  /* Adjust size */
            font-weight: bold;
            text-align: center;
            
        }
    </style>
""", unsafe_allow_html=True)

# Colorful, smaller title
st.markdown('<h1 class="custom-title">Time Series Forecasting System</h1>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        This app allows you to upload a time series dataset, clean the data, visualize decomposition 
        and forecast future demand or Sales using models like Statistical , ML and DL.
    </div>
""", unsafe_allow_html=True)

# Custom CSS for menu with increased tab width
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 20px 20px;  /* Increased padding to make tabs wider */
        margin-right: 2px;
        font-weight: normal;
        font-size: 30px;
        transition: all 0.5s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: red;
    }
    .stTabs [data-baseweb="tab"]:active {
        color: red;
    }
    .stTabs [data-baseweb="tab"]:focus {
        outline: none;
    }
    </style>
""", unsafe_allow_html=True)

mui_table_css = """
.MuiTable-root {
    border: 2px solid #454545; /* Green border around the table */
    border-collapse: collapse; /* Ensures borders are merged */
    width: 100%; /* Full width */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
}

.MuiTableCell-head {
    background-color: #4863A0; /* Blue header background */
    color: white; /* White text for headers */
    font-weight: bold;
    padding: 12px; /* Padding for header cells */
    border-bottom: 2px solid #454545; /* Green border below headers */
}

.MuiTableCell-body {
    padding: 10px; /* Padding for body cells */
    border-bottom: 1px solid #454545; /* Light border between rows */
}

.MuiTableRow:nth-child(odd) {
    background-color: #f9f9f9; /* Light gray for odd rows */
}

.MuiTableRow:nth-child(even) {
    background-color: #ffffff; /* White for even rows */
}

.MuiTableRow:hover {
    background-color: #FFEB3B; /* Yellow hover effect */
    transition: background-color 0.3s ease; /* Smooth transition */
}

.MuiTableRow:last-child .MuiTableCell-body {
    border-bottom: none; /* Remove border from the last row */
}
"""

# Horizontal menu navigation
menu = st.tabs([":material/database: Data Display", ":material/build: Missing Handling",":material/carpenter: Anomaly Handling", ":material/analytics: Statistical Analysis", ":material/trending_up: Forecasting",":material/bar_chart: Dashboard"])
st.sidebar.image("images/logo-bits.png",  use_container_width=True)
# Path to the CSV template file (update with the correct path)
file_path = "input/data_template.csv"

# Read the file as bytes
with open(file_path, "rb") as file:
    file_bytes = file.read()

# Sidebar Download Button
st.sidebar.header("Sample data template")
st.sidebar.download_button(
    label="📥 Download Data Template",
    data=file_bytes,
    file_name="data_template.csv",
    mime="text/csv"
)


st.sidebar.header("🔼 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file containing time series data", type=["csv"])

if uploaded_file:
    # Load and display the data
    #st.rerun() 
    data = pd.read_csv(uploaded_file)
    for col in data.select_dtypes(include=["int64", "object"]):  # Check for potential numbers stored as text
        try:
            data[col] = data[col].astype(float)
        except ValueError:
            pass  # Skip non-convertible columns
            
    st.sidebar.success("Data uploaded successfully!")
    # Sidebar: Data selection
    st.sidebar.header("📅 Column Selection")
    date_column = st.sidebar.selectbox("Select the Date Column", data.columns[0])
    
    if(len(data.columns)>4):
        region_column=data.columns[1]
        sku_column=data.columns[2]
        type_column=data.columns[3]
        regions = data[region_column].unique()
        #skus = data[sku_column].unique()        
        types = data[type_column].unique()
        
        selected_region = st.sidebar.selectbox("Select Region", regions)
        filtered_skus = data[data[region_column] == selected_region][sku_column].unique()
        selected_sku = st.sidebar.selectbox("Select SKU", filtered_skus)
        #selected_sku = st.sidebar.selectbox("Select SKU", skus)
        selected_type = st.sidebar.selectbox("Select Type", types)
        data = data[(data[region_column] == selected_region) & (data[sku_column] == selected_sku) & (data[type_column] == selected_type)]
        value_column = st.sidebar.selectbox("Select the Sales Column", data.columns[4])
        value_columns=[]
        if(len(data.columns)>5):
            value_column_list=data.columns[5:len(data.columns)]            
            value_columns = st.sidebar.multiselect("Select Features Column", value_column_list)
            data=data[[date_column,region_column,sku_column,type_column,value_column]+value_columns]
        else:
            data=data[[date_column,region_column,sku_column,type_column,value_column]+value_columns]
    else:
        value_column = st.sidebar.selectbox("Select the Sales Column", data.columns[len(data.columns)-1])
    
    # Process data
    data[date_column] = pd.to_datetime(data[date_column], format="%d-%m-%Y",errors="coerce")
    
    data = data.set_index(date_column).sort_index()
    st.session_state['data'] = data
    time_freq = pd.infer_freq(data.index)    
    st.session_state['time_freq'] = time_freq
    
# Page 1: Upload Data
with menu[0]:
    st.header("Data Display & Visualization")
    if st.session_state['data'] is not None:
        data = st.session_state['data'].copy()      
        
        temp = data.copy()
        
        temp = temp.reset_index()
        temp[date_column] = temp[date_column].astype(str)
        
        st_mui_table(
            temp,   
            paginationSizes=[5,10, 25],        
            size="small", 
            customCss=mui_table_css,
            #padding="checkbox",
            showHeaders=True, 
            key="mui_table" , 
            detailsHeader="Details",
            detailColNum=1,
            enablePagination=True,
            showIndex=False
        )
            
        
# Page 2: 🛠️ Missing Handling
with menu[1]:
    if st.session_state['data'] is not None:
        data = st.session_state['data'].copy()  
        original = data.copy()
        
        if data.isnull().sum().sum() > 0:
            #st.warning(f"🚨 Missing values detected: {data.isnull().sum().sum()}")
            missing_strategy = st.selectbox("Select Imputation Method", 
                                            ["Mean", "Median", "Mode","Forward Fill", "Backward Fill","Linear Interpolation",
                                             "K-Nearest","Best Fill","Drop Rows"])
            if missing_strategy == "Drop Rows":
                miss_handled_data = data.dropna()
            elif missing_strategy == "Mean":
                miss_handled_data = mean_case(data)
            elif missing_strategy == "Median":
                miss_handled_data = median_case(data)
            elif missing_strategy == "Mode":
                miss_handled_data = mode_case(data)
            elif missing_strategy == "Forward Fill":
                miss_handled_data = forward_fill_case(data)
            elif missing_strategy == "Backward Fill":
                miss_handled_data = backward_fill_case(data)
            elif missing_strategy == "Linear Interpolation":
                miss_handled_data = linear_interpolation_case(data)
            elif missing_strategy == "K-Nearest":
                miss_handled_data = k_nearest_case(data)
            elif missing_strategy == "Best Fill":
                miss_handled_data=choose_best_method(data)   
                
                  
            col1, col2 = st.columns(2, gap="small")
            with col1:
                # Visualize Original Data with Missing Values Marked in Red
                st.subheader("Original Data")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(original.index, original[value_column], label="Original Data", color="blue")
                ax.legend()
                ax.set_ylabel(value_column)
                ax.set_xlabel("Date")
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)
            with col2:
                # Visualize Original Data with Missing Values Marked in Red
                st.subheader("Missing handled data")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(miss_handled_data.index, miss_handled_data[value_column], label="Filled Data", color="green")           
                ax.legend()
                ax.set_ylabel(value_column)
                ax.set_xlabel("Date")
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig) 
                
            st.session_state['missing_handled_data'] = miss_handled_data
            

        else:
            missing_strategy = st.selectbox("Select Imputation Method", 
                                            ["Mean", "Median", "Mode","Forward Fill", "Backward Fill","Linear Interpolation",
                                                "Polynomial Interpolation","K-Nearest","Best Fill","Drop Rows"],disabled=True)  

            # Visualize Original Data with Missing Values Marked in Red
            st.subheader("Original Data")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(data.index, data[value_column], label="Original Data", color="blue")
            ax.legend()
            ax.set_ylabel(value_column)
            ax.set_xlabel("Date")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            
            st.session_state['missing_handled_data'] = data

 
    else:
        st.info("📂 Please upload data in the 'Upload Data' tab.")
        
# Page 3: 🧹 Anomaly Handling
with menu[2]:
    if st.session_state['missing_handled_data'] is not None:
        data = st.session_state['missing_handled_data'].copy()  
        # Outlier detection and handling
        q1 = data[value_column].quantile(0.25)
        q3 = data[value_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[value_column] < lower_bound) | (data[value_column] > upper_bound)]        
        lower_bound_list=[lower_bound for x in data[value_column]]
        upper_bound_list=[upper_bound for x in data[value_column]]
        temp_data=st.session_state['missing_handled_data'].copy()
        temp_data['Upper']=upper_bound_list
        temp_data['Lower']=lower_bound_list
            
        if not outliers.empty:
            #st.warning(f"🚨 Detected {len(outliers)} outliers.")
            outlier_strategy = st.selectbox("Select Outlier Handling Strategy", ["Do Nothing","IQR Outliers", "Cap Outliers"])
            if outlier_strategy == "IQR Outliers":
                data = data[(data[value_column] >= lower_bound) & (data[value_column] <= upper_bound)]
            elif outlier_strategy == "Cap Outliers":
                data[value_column] = np.clip(data[value_column], lower_bound, upper_bound)
            elif outlier_strategy == "Do Nothing":
                data[value_column] = data[value_column]

        # Save cleaned data to session state
        st.session_state['cleaned_data'] = data
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot using Seaborn for a more visually appealing style
        sns.lineplot(data=data, x=data.index, y=value_column, ax=ax, label="Data", color="blue")
        sns.lineplot(data=temp_data, x=temp_data.index, y="Upper", ax=ax, label="Upper Bound", color="red")
        sns.lineplot(data=temp_data, x=temp_data.index, y="Lower", ax=ax, label="Lower Bound", color="orange")

        # Set title and labels
        #ax.set_title("Cleaned Data", fontsize=12, fontweight="bold")
        ax.set_ylabel(value_column, fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        # Customize legend
        ax.legend(loc="upper left", fontsize=10)
        # Grid styling
        ax.grid(visible=True, linestyle="--", alpha=0.6)
        # Center the figure in Streamlit
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("📂 Please upload data in the 'Upload Data' tab.")
        
# Page 4: Statistical Analysis Visualization
with menu[3]:
    if st.session_state['cleaned_data'] is not None:
        data = st.session_state['cleaned_data'].copy()
        
        time_freq = st.session_state['time_freq']
        if(time_freq=='MS'):
            data_period=12
        elif(time_freq=='D'):
            data_period=7
        else:
            data_period=1      
        
        contains_zero = (data[value_column] <= 0).any().any()   
        model_flag=True             
        if(contains_zero==False):                    
            model_flag=False
        else:
            model_flag=True
                        
        if(model_flag==True):
            forecast1,model_fit1 = Holt_Winters_Model(3,data[value_column],'add',data_period)
        else:
            forecast1,model_fit1 = Holt_Winters_Model(3,data[value_column],'mul',data_period)
        
        alpha =round(model_fit1.params['smoothing_level'],4)
        beta = round(model_fit1.params['smoothing_trend'],4)
        gamma = round(model_fit1.params['smoothing_seasonal'],4)
        
        step=1  
        
        forecast2,level,trend,season,rmse=additive(data[value_column],data_period,step,alpha, beta, gamma)
        forecast_valu=0
        level_fraction=0
        trend_fraction=0
        season_fraction=0
        k=1      
        for i in range(len(forecast2)): 
            if(k<step):
                timeid=k
                k=k+1
            elif(k==step):
                timeid=k
                k=1            
            l=level[i]                
            if(beta<=0.0001):
                t=trend[i]*beta
            else:
                t=trend[i]            
            s=season[i]            
            f=forecast2[i]
            forecast_valu=round(f)
            level_fraction=round(l)
            trend_fraction=round(t)
            season_fraction=round(s)
      
        st.markdown("#### 📊 𝛂, β, and γ Parameters & Forecast Value Bifurcation 🔀 based on Holt-Winters Model Forecast")
        
        colw1, colw2 = st.columns([0.45,0.55], gap="small")  
        with colw1:
            col11, col12 ,col13 = st.columns(3)                
            col11.metric(label="α parameter for Level", value=alpha,border=True)
            col12.metric(label="β parameter for Trend", value=beta,border=True)
            col13.metric(label="γ parameter for Gama", value=gamma,border=True)  
        with colw2:                   
            col14,col15,col16,col17= st.columns(4)
            col14.metric(label="Forecsated Qty (FC)", value=forecast_valu,border=True)
            col15.metric(label="Level fraction of FC", value=level_fraction,border=True)
            col16.metric(label="Trend fraction of FC", value=trend_fraction,border=True)
            col17.metric(label="Season fraction of FC", value=season_fraction,border=True) 
        #st.markdown("---") 
        decomposition = seasonal_decompose(data[value_column], model="Additive", period=data_period)  
               # Plot decomposition
        fig, axes = plt.subplots(3, 1, figsize=(12, 4))
        # axes[0].plot(data[value_column], label="Observed", color="blue")
        # axes[0].set_title("Observed")
        axes[0].plot(decomposition.trend, label="Trend", color="green")
        axes[0].set_title("Trend")
        axes[1].plot(decomposition.seasonal, label="Seasonality", color="orange")
        axes[1].set_title("Seasonality")
        axes[2].plot(decomposition.resid, label="Residuals", color="red")
        axes[2].set_title("Residuals")
        fig.tight_layout()

        st.session_state['decomposition_graph'] = fig
        st.pyplot(fig)  
        data2=data.reset_index()
        
        col1, col2 = st.columns(2, gap="small")  
        with col1:
            # Autocorrelation Plot (ACF)
            st.write("### Autocorrelation Function (ACF) Plot")
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_acf(data2[value_column], lags=12, ax=ax)
            ax.set_title("Autocorrelation Function (ACF)")
            st.pyplot(fig) 
        with col2:
            # Partial Autocorrelation Plot (PACF)
            st.write("### Partial Autocorrelation Function (PACF) Plot")
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_pacf(data2[value_column], lags=12, ax=ax)
            ax.set_title("Partial Autocorrelation Function (PACF)")
            st.pyplot(fig)
          
        col11, col22 = st.columns(2, gap="small") 
        # Display highest and lowest sales
        top_sale = data2[data2[value_column] == data2[value_column].max()]
        low_sale = data2[data2[value_column] == data2[value_column].min()]
        with col11:            
            # Moving Average
            st.write("### Moving Average of Sales Quantity")
            data2['MA_3'] = data2[value_column].rolling(window=3).mean()
            fig3 = px.line(data2, x=date_column, y=[value_column, 'MA_3'],  markers=True)
            st.plotly_chart(fig3)            
        with col22:
            # Histogram
            st.write("### Distribution of Sales Quantity")
            fig, ax = plt.subplots(figsize=(11, 7))
            sns.histplot(data2[value_column], bins=15, kde=True, ax=ax)
            ax.set_title("Sales Quantity Distribution")
            ax.set_xlabel("Quantity Sold")
            st.pyplot(fig)

    else:
        st.info("📂 Please clean data in the 'Data Cleaning' tab.")
        

# Page 5: Forecasting
with menu[4]:
    if st.session_state['cleaned_data'] is not None:
        data = st.session_state['cleaned_data'].copy()
        data2 = data.reset_index()
        end_date=list(data2[date_column])[len(data2[date_column])-1]                  
        end_date_str = end_date.strftime('%d-%m-%Y')  # Convert Timestamp to string
        end_date_obj = datetime.strptime(end_date_str, '%d-%m-%Y') 
        time_freq = st.session_state['time_freq']
        if(time_freq=='MS'):
            data_period=12
        elif(time_freq=='D'):
            data_period=7
        else:
            data_period=1 
            
        suggested_models=[]                             
        sel_model= forecast_model_selection_algorithm(data2[[date_column,value_column]+value_columns],data_period)   
        st_models=sel_model["STATS"]   
        ml_models=sel_model["ML"]
        dl_models=sel_model["DL"]   
        if(len(value_columns)>0):
            suggested_models=ml_models+dl_models
        else:
            suggested_models=st_models+ml_models+dl_models
        st.session_state['suggested_models']=suggested_models        
        if(suggested_models!=[]):
            if(len(value_columns)>0):
                st.success(f"##### **✔️ Suggested Models (Multivariate) :** {', '.join(suggested_models)} (Based on model selection algorithm)")
            else:
                st.success(f"##### **✔️ Suggested Models (Univariate) :** {', '.join(suggested_models)} (Based on model selection algorithm)")
           
        model_type=[]      
        if(len(value_columns)==0):
            col1, col2 ,col3 ,col4,col5 = st.columns(5, gap="small")
            with col1:
                if(time_freq=='MS'):
                    forecast_periods = st.slider("Forecast Periods (Months)", 1, 6, 3)
                elif(time_freq=='D'):
                    forecast_periods = st.slider("Forecast Periods (Days)", 1, 30,15)
                else:
                    forecast_periods = st.slider("Forecast Periods (Years)", 1, 12, 6)            
            with col2:
                model_type1 = st.multiselect("Statistics Model", 
                ["Holt-Winters", "ARIMA", "SARIMA"])
            with col3:
                model_type2 = st.multiselect("ML Model", 
                ["Random Forest","SV Regression","XGboost Regression","Voting Regression"])
            with col4:
                model_type3 = st.multiselect("DL Model",["RNN","CNN","PROPHET","LSTM"])
            with col5:
                model_type4 = st.multiselect("Transformer Model",["Lagllama","Lagllama_Tuned"])
            
            model_type=model_type1+model_type2+model_type3+model_type4
        else:
            col1, col2 ,col3  = st.columns(3, gap="small")
            with col1:
                if(time_freq=='MS'):
                    forecast_periods = st.slider("Forecast Periods (Months)", 1, 6, 3)
                elif(time_freq=='D'):
                    forecast_periods = st.slider("Forecast Periods (Days)", 1, 30,15)
                else:
                    forecast_periods = st.slider("Forecast Periods (Years)", 1, 12, 6)                
            
            with col2:
                model_type2 = st.multiselect("ML Model", 
                ["Random Forest","XGboost Regression","Voting Regression"])
            with col3:
                model_type3 = st.multiselect("DL Model",["RNN","CNN","LSTM"])           
            model_type=model_type2+model_type3
        
        change_flag=False
        if(st.session_state['forecast_periods'] == forecast_periods):
            change_flag=False
        else:
            change_flag=True
            st.session_state['forecast_periods'] = forecast_periods
            
        def run_model():
            with st.spinner(f'Executing the model {model_type}, please wait...'):
                model_selected = []
                forecasted_all_results = []
                model_fit_all_results = []
                fitted_data_all_results = []
                lag_mae,lag_mape,lag_rmse=0,0,0
                tlag_mae,tlag_mape,tlag_rmse=0,0,0
                model_flag=True            
                if  "Holt-Winters" in model_type:
                    contains_zero = (data[value_column] <= 0).any().any()                
                    if(contains_zero==False):                    
                        forecast,model_fit = Holt_Winters_Model(forecast_periods,data[value_column],'mul',data_period)
                        model_flag=False
                        forecasted_all_results.append(forecast)
                        fitted_forecast =model_fit.fittedvalues 
                        fitted_data_all_results.append(fitted_forecast)
                        model_selected.append('HWMUL')
                    else:
                        #st.write("Data contains zero values")
                        forecast,model_fit = Holt_Winters_Model(forecast_periods,data[value_column],'add',data_period)
                        model_flag=True
                        forecasted_all_results.append(forecast)
                        fitted_forecast =model_fit.fittedvalues 
                        fitted_data_all_results.append(fitted_forecast)
                        model_selected.append('HWADD')

                if "ARIMA" in model_type:
                    best_pqd=evaluate_arima_order(data[value_column])
                    #best_pqd=(1,2,1)
                    forecast,model_fit=ARIMA_Model(forecast_periods,data[value_column],best_pqd)
                    forecasted_all_results.append(forecast)
                    fitted_forecast =model_fit.fittedvalues 
                    fitted_data_all_results.append(fitted_forecast)
                    model_selected.append('ARIMA')
                if "SARIMA" in model_type:
                    best_pdq,best_seasonal_pdq=evaluate_sarima_order(data[value_column],data_period)                
                    forecast,model_fit = SARIMA_Model(forecast_periods,data[value_column],best_pdq,best_seasonal_pdq) 
                    forecasted_all_results.append(forecast)
                    fitted_forecast =model_fit.fittedvalues 
                    fitted_data_all_results.append(fitted_forecast)
                    model_selected.append('SARIMA')                    
                if "Random Forest" in model_type:
                    forecast,model_fit,fitted_data = RFR_Model(forecast_periods,end_date_obj,data,value_column,date_column,value_columns)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('Random Forest')
                if "SV Regression" in model_type:
                    forecast,model_fit,fitted_data = SVR_Model(forecast_periods,end_date_obj,data,value_column,date_column)  
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('SV Regression')
                if "XGboost Regression" in model_type:
                    forecast,model_fit,fitted_data = XGBR_Model(forecast_periods,end_date_obj,data,value_column,date_column,value_columns)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('XGboost Regression')
                if "Voting Regression" in model_type:
                    forecast,model_fit,fitted_data = Voting_Model(forecast_periods,end_date_obj,data,value_column,date_column,value_columns)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('Voting Regression')
                if  "LSTM" in model_type:
                    forecast,model_fit,fitted_data = LSTM_Model(forecast_periods,end_date_obj,data,value_column,date_column,value_columns)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('LSTM')
                if  "Lagllama" in model_type:
                    forecast,model_fit,fitted_data,lag_mae,lag_mape,lag_rmse = Lagllama_Model(forecast_periods,end_date_obj,data,value_column,date_column,time_freq)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('Lag-llama')
                if  "Lagllama_Tuned" in model_type:
                    forecast,model_fit,fitted_data,tlag_mae,tlag_mape,tlag_rmse = Lagllama_Model_Tuned(forecast_periods,end_date_obj,data,value_column,date_column,time_freq)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('Lag-llama-Tuned')                    
                if  "PROPHET" in model_type:
                    forecast,model_fit,fitted_data = PROPHET_Model(forecast_periods,end_date_obj,data,value_column,date_column,time_freq)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('PROPHET')                
                if  "CNN" in model_type:
                    forecast,model_fit,fitted_data = CNN_Model(forecast_periods,end_date_obj,data,value_column,date_column)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('CNN')
                if  "RNN" in model_type:
                    forecast,model_fit,fitted_data = RNN_Model(forecast_periods,end_date_obj,data,value_column,date_column)
                    forecasted_all_results.append(forecast)
                    fitted_data_all_results.append(fitted_data)
                    model_selected.append('RNN')   
                
                return forecasted_all_results, fitted_data_all_results, model_selected ,lag_mae,lag_mape,lag_rmse,tlag_mae,tlag_mape,tlag_rmse
        if change_flag==False:    
            if st.session_state.models is not None:
                if model_type == st.session_state.models:
                    forecasted_all_results, fitted_data_all_results, model_selected,lag_mae,lag_mape,lag_rmse,tlag_mae,tlag_mape,tlag_rmse = st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse

                    pass
                else:
                    st.session_state.models = model_type
                    st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse = run_model()
                    forecasted_all_results, fitted_data_all_results, model_selected, lag_mae,lag_mape,lag_rmse,tlag_mae,tlag_mape,tlag_rmse=st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse

            else:
                st.session_state.models = model_type     
                st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse = run_model() 
                forecasted_all_results, fitted_data_all_results, model_selected,lag_mae,lag_mape,lag_rmse,tlag_mae,tlag_mape,tlag_rmse = st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse
        else:
            st.session_state.models = model_type
            st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse = run_model()
            forecasted_all_results, fitted_data_all_results, model_selected, lag_mae,lag_mape,lag_rmse,tlag_mae,tlag_mape,tlag_rmse=st.session_state.fcr, st.session_state.fdr, st.session_state.ms,st.session_state.lmae,st.session_state.lmape,st.session_state.lrmse,st.session_state.tlmae,st.session_state.tlmape,st.session_state.tlrmse

                
        final_result_df = pd.DataFrame()
        for forecast, fitted_forecast, m_name in zip(forecasted_all_results, fitted_data_all_results, model_selected):                
            combined_data = data.iloc[:, [0, 1, 2,3]].copy() 
            a, s, t, _= combined_data.iloc[0].values
            
            an, sn, tn, _ = combined_data.columns[0:4]
            combined_data["Model"] = m_name
            forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=forecast_periods, freq=time_freq)
            forecast=np.array(forecast)
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"]) 
            forecast_df[an] = a
            forecast_df[sn] = s
            forecast_df[tn] = t
            
            forecast_df["Model"] = m_name
            fitted_forecast_value=np.array(fitted_forecast)
            combined_data["Fitted"] = fitted_forecast_value
            combined_data = pd.concat([combined_data, forecast_df],axis=0)
            combined_data["Fitted"] = np.append(fitted_forecast, [None] * forecast_periods)
        
            combined_data.index.name = date_column            
            if(m_name=="Lag-llama"):
                mae=lag_mae
                mape=lag_mape
                rmse=lag_rmse
                r2 = 1-(mape/100)                
            elif(m_name=="Lag-llama-Tuned"):
                mae=tlag_mae
                mape=tlag_mape
                rmse=tlag_rmse
                r2 = 1-(mape/100)
            else:
                # Calculate Metrics (Adding as Extra Columns)
                mae = mean_absolute_error(data[value_column], fitted_forecast)
                mape = cal_mape(data[value_column],fitted_forecast,len(fitted_forecast))
                rmse = np.sqrt(mean_squared_error(data[value_column], fitted_forecast))
                r2 = r2_score(data[value_column], fitted_forecast)

            # Append metrics as extra columns to each row
            combined_data["MAE"] = mae
            combined_data["MAPE"] = mape
            combined_data["RMSE"] = rmse
            combined_data["R²"] = r2
            if(mape<=20):
                remark="Good"
            elif(mape>20 and mape<=30):
                remark="Satisfactory"
            else:
                remark="Unsatisfactory"
            combined_data["Remark"] = remark
            
            q1 = data[value_column].quantile(0.25)
            q3 = data[value_column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            if(lower_bound<0):
                lower_bound=min(data[value_column])
            combined_data["Upper Bound"] = upper_bound
            combined_data["Lower Bound "] = lower_bound

            final_result_df = pd.concat([final_result_df,combined_data])
            
        download_final_result_df = pd.DataFrame() 
        combined_results = [] 
        for forecast, fitted_forecast, m_name in zip(forecasted_all_results, fitted_data_all_results, model_selected): 
            combined_data = data.iloc[:len(forecast), [0, 1, 2]].copy()
            forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=forecast_periods, freq=time_freq)
            forecast_df = pd.DataFrame({"Forecast Qty": np.round(np.array(forecast),0)}, index=forecast_index)
            combined_data["Model"] = m_name
            
            if(m_name=="Lag-llama"):
                mae=lag_mae
                mape=lag_mape
                rmse=lag_rmse
                r2 = 1-(mape/100)
            elif(m_name=="Lag-llama-Tuned"):
                mae=tlag_mae
                mape=tlag_mape
                rmse=tlag_rmse
                r2 = 1-(mape/100)
            else:
                # Calculate Metrics
                mae = round(mean_absolute_error(data[value_column], fitted_forecast),2)
                mape = round(cal_mape(data[value_column], fitted_forecast, len(fitted_forecast)),2)
                rmse = round(np.sqrt(mean_squared_error(data[value_column], fitted_forecast)),2)
                r2 = round(r2_score(data[value_column], fitted_forecast),2)
                
            # Create DataFrames for metrics
            mae_df = pd.DataFrame({"MAE": [mae] * forecast_periods}, index=forecast_index)
            mape_df = pd.DataFrame({"MAPE": [mape] * forecast_periods}, index=forecast_index)
            rmse_df = pd.DataFrame({"RMSE": [rmse] * forecast_periods}, index=forecast_index)
            r2_df = pd.DataFrame({"R²": [r2] * forecast_periods}, index=forecast_index)

            combined_data = combined_data.iloc[:forecast_periods].set_index(forecast_index)
            combined_data = pd.concat([combined_data, forecast_df, mae_df, mape_df, rmse_df, r2_df], axis=1)
            combined_results.append(combined_data)          
        if(combined_results==[]):
            combined_results.append(pd.DataFrame(columns=["Model","MAE","MAPE","RMSE","R²","Remark"])) 
        else:
            download_final_result_df = pd.concat(combined_results)
            download_final_result_df=download_final_result_df.reset_index().rename(columns={"index": "Date"})
            download_final_result_df["Date"] = download_final_result_df["Date"].dt.strftime("%d-%m-%Y")

            
        model_colors = sns.color_palette("husl", len(model_selected))
        st.markdown("#### 📈 Forecast Visualization and Performance Metrics for Selected Models") 
        col1, col2 = st.columns(2, gap="small")             
        with col1: 
            
            fig, ax = plt.subplots(figsize=(18,11))
            ax.plot(data.index, data[value_column], label="Actual", color="black", linewidth=2)
            for idx, (forecast, fitted_forecast, m_name) in enumerate(zip(forecasted_all_results, fitted_data_all_results, model_selected)):
                color = model_colors[idx]  # Assign a unique color per model                    
                ax.plot(data.index, fitted_forecast, color=color, label=f"{m_name} (Fitted)", alpha=0.8,linewidth=2)                    
                forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=forecast_periods, freq=time_freq)
                ax.plot(forecast_index, forecast, linestyle="solid", color=color, label=f"{m_name} (Forecast)", linewidth=2)
                
            ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="black")
            ax.set_ylabel(value_column, fontsize=12)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_title("Visualization Time Series Forecasting", fontsize=16, fontweight="bold", color="darkblue")
            ax.grid(visible=True, linestyle="--", alpha=0.6)
            st.pyplot(fig)    
               
        # Metrics table in the second column
        with col2:
            metrics_list = []
            for forecast, fitted_forecast, m_name in zip(forecasted_all_results, fitted_data_all_results, model_selected):
                mae = mean_absolute_error(data[value_column], fitted_forecast)
                mape = cal_mape(data[value_column], fitted_forecast, len(fitted_forecast))
                rmse = np.sqrt(mean_squared_error(data[value_column], fitted_forecast))
                r2 = r2_score(data[value_column], fitted_forecast)
                if(m_name=="Lag-llama"):
                    mae=lag_mae
                    mape=lag_mape
                    rmse=lag_rmse
                    r2 = 1-(mape/100)
                if(m_name=="Lag-llama-Tuned"):
                    mae=tlag_mae
                    mape=tlag_mape
                    rmse=tlag_rmse
                    r2 = 1-(mape/100)
                
                if(mape<=20):
                    remark="Good"
                elif(mape>20 and mape<=30):
                    remark="Satisfactory"
                else:
                    remark="Unsatisfactory"
                    
                metrics_list.append([m_name, mae, mape, rmse, r2,remark])

            metrics_df = pd.DataFrame(metrics_list, columns=["Model", "MAE", "MAPE", "RMSE", "R²","Remark"])

            # Apply styling to enhance display
            def highlight_max_r2(s):
                is_max = s == s.max()
                return ['background-color: #FFEB3B; color: black; font-weight: bold;' if v else '' for v in is_max]

            def highlight_min_mape(s):
                is_min = s == s.min()
                return ['background-color: #4CAF50; color: white; font-weight: bold;' if v else '' for v in is_min]

            styled_df = metrics_df.style.format(
                {"MAE": "{:.2f}", "MAPE": "{:.2f}%", "RMSE": "{:.2f}", "R²": "{:.2f}"}
            ).apply(highlight_max_r2, subset=["R²"]).apply(highlight_min_mape, subset=["MAPE"])

            # Apply global table styles
            styled_df = styled_df.set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#1565C0'), 
                                                    ('color', 'white'), ('font-weight', 'bold'), 
                                                    ('text-align', 'center'), ('border', '2px solid #1565C0')]},
                {'selector': 'tbody td', 'props': [('border', '2px solid #ddd'), ('padding', '10px'), 
                                                    ('text-align', 'center'), ('font-size', '14px')]},
                {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f1f8e9'), ('border', '2px solid #ddd')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#e3f2fd'), ('border', '2px solid #ddd')]},
                {'selector': 'table', 'props': [('border', '2px solid #1565C0'), ('border-collapse', 'collapse')]}
            ])
            
            # Display enhanced dataframe with markdown style
            st.markdown(
                """
                <style>
                .stDataFrame { 
                    width: 100% !important; 
                    max-width: 100%; 
                    border: 2px solid #9a9a9a; 
                    padding: 1px;
                    overflow: hidden;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.dataframe(styled_df, hide_index=True, height=255, width=700)            
            
            ## Save result into data base
            #if st.button("💾 Save Results to Database"):
            run_type="Direct"   ## Direct / Train-Test
            eval_type="Rolling" ## Rolling / Sliding
            if(len(model_type)>0):
                save_result_db(final_result_df,run_type,eval_type)             
    
            
            # Store Excel file in session state
            excel_buffer = io.BytesIO()
            download_final_result_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            # Provide download button without reloading
            st.download_button(
                label="Download Forecast Result",
                data=excel_buffer,
                file_name="forecast_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
                
    else:
        st.info("📂 Please clean data in the 'Data Cleaning' tab.")

# Page 6: Dashboard
with menu[5]:
    # Embed the Power BI dashboard using an iframe
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=d8c9e3ad-a2ec-4105-9866-8adc0d176d3c&autoAuth=true&ctid=01106cca-5321-44ee-a197-bb9dfabbb478"

    iframe_html = f"""
    <iframe 
        src="{power_bi_url}" 
        style="width: 100%; height: 80vh; border: none;"
        allowfullscreen>
    </iframe>
    """

    st.components.v1.html(iframe_html, height=768)