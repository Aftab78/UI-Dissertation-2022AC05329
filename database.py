import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import math 
from datetime import datetime
import pyodbc
def get_connection_string():
    # Define database connection parameters
    server = '4.240.93.106'  # e.g., 'localhost' or 'server\instance'
    database = 'FORECAST_DEV'
    username = 'wilp'
    password = 'wilp'
    
    conn_str = 'DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password+';Trusted_Connection=no'
       
    return conn_str
    
def create_save_result_table():
    conn_str = get_connection_string()
    # Establish connection to SQL Server
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Create the table if it does not exist
    cursor.execute('''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='forecast_result' AND xtype='U')
    CREATE TABLE forecast_result (
        Date DATE,
        Region NVARCHAR(255),
        SKU NVARCHAR(255),
        Type NVARCHAR(255),
        Sales_Qty FLOAT,
        Model NVARCHAR(255),
        Fitted FLOAT,
        Forecast FLOAT,
        MAE FLOAT,
        MAPE FLOAT,
        RMSE FLOAT,
        R2 FLOAT
    )''')
    conn.commit()
    conn.close()
    cursor.close()

def create_scenario(scenid,scen_des):  
    
        
    ## First  delete old result 
    conn_str = get_connection_string()
    
    data_list=[[scenid,scen_des]]
    # Establish connection to SQL Server
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()    
    # Insert new data
    cursor.executemany("scenario_insert ?,?", data_list)
    conn.commit()
    conn.close()
    
    print("Scenario created successfully!")
    
def save_result_db(df,run_type,eval_type):
    df = df.reset_index()
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%Y-%m-%d",errors="coerce")
    #df[df.columns[0]] = df[df.columns[0]].dt.date
    data = [tuple(row) for row in df.values]   
    data_list=[]


    # Get current date and time
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d%b%Y_%I_%p")
    scienid = "S_"+formatted_time
    print("Scenario Id : ",scienid)
    create_scenario(scienid,scienid)
    
    for item in data:
        date=item[0]
        area=item[1]
        product=item[2]
        type=item[3]
        if(item[4]=="" or item[4]==None or math.isnan(float(item[4])) ):
            sales_qty=None
        else:
            sales_qty=item[4]
            
        model=item[5]
        if(item[6]=="" or item[6]==None or math.isnan(float(item[6]))):
            fitted=None
        else:
            fitted=item[6]
        
        
        if(item[7]=="" or item[7]==None  or math.isnan(float(item[7]))):
            forecast=None
        else:
            forecast=item[7]
       
        mae=item[8]
        mape=item[9]
        rmse=item[10]
        r2=item[11]
        remark=item[12]   
        upper_boud=item[13]    
        lower_boud=item[14]      
        #print([date,area,product,type,sales_qty,model,fitted,forecast,mae,mape,rmse,r2,remark])
        data_list.append([date,area,product,type,sales_qty,model,fitted,forecast,mae,mape,rmse,r2,remark,upper_boud,lower_boud,scienid,run_type,eval_type])
        
    ## First  delete old result 
    conn_str = get_connection_string()
    
    # Establish connection to SQL Server
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()    
    # Insert new data
    cursor.execute("delete from forecast_result where scen_id ="+ "'"+scienid+"'")
    conn.commit()
    conn.close()
    
    # Establish connection to SQL Server
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()    
    # Insert new data
    cursor.executemany("forecast_insert ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?", data_list)
    conn.commit()
    conn.close()
    
    print("Data inserted successfully!")