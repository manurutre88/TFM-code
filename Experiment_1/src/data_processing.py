import pandas as pd
import os

BASE_DIR = os.getcwd()
data_file_path = os.path.join(BASE_DIR,'data','dataset_auga1','train_dataset','train_ord.csv')

# ////////////////////////////////////////////////////////////////////////////////////////////

def process_nan(df, modo=1):
    if modo == 0:
        return df
    elif modo == 1:
        return df.dropna()
    elif modo == 2:
        return df.fillna(0)
    
# ////////////////////////////////////////////////////////////////////////////////////////////

def import_data(modo=1):

    df = pd.read_csv(data_file_path, index_col=0, header=0)
    df = process_nan(df, modo)
    return df

# ////////////////////////////////////////////////////////////////////////////////////////////

def group_data(df):

    df_daily = df
    df_daily.columns = pd.to_datetime(df_daily.columns, format='%d-%m-%Y')
    df_daily = df_daily.T

    df_weekly = df_daily.resample('W').sum().drop(df_daily.resample('W').sum().index[-1], axis=0)
    df_weekly.index = [f"Semana_{i+1}" for i in range(len(df_weekly.index))]

    df_monthly = df_daily.resample('ME').sum()
    df_monthly.index = [f"Mes_{i+1}" for i in range(len(df_monthly.index))]

    return df_weekly.T, df_monthly.T

# ////////////////////////////////////////////////////////////////////////////////////////////
