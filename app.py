import streamlit as st
import pandas as pd
from pandas_datareader.stooq import StooqDailyReader
import numpy as np
from datetime import datetime
import pickle

import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet

f = open('./prophet-model.pkl', 'rb') 
m = pickle.load(f)

start = datetime(1999, 4, 6)
end = datetime.today()


@st.cache
def load_data(ticker):
    result = StooqDailyReader(ticker, start, end)
    data = result.read()
    data = data.iloc[::-1]
    data.reset_index(inplace=True)
    return data

# progress_bar = st.sidebar.progress(0)
data = load_data('9202.JP')

# progress_bar.progress(1)

def run():

    st.title('Simple Stock Prediction with Prophet')
    st.subheader('Prediction On ANA Japan Airline')

    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    # m = Prophet()
    # m.fit(df_train)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    
    # st.dataframe(forecast.tail(30))
    
    # df_final = forecast.tail(30)
    # df_final = df_final[['ds', 'yhat']]
    # df_final = df_final.set_index('ds')
    
    # st.dataframe(df_train)
    # st.dataframe(df_final)

    # merge 

    # st.line_chart(df_final)
    # progress_bar.empty()

    temp = forecast[['ds', 'yhat']]
    df_merge = pd.merge(df_train, temp, on='ds')

    df_merge = df_merge.set_index('ds')

    # st.dataframe(df_merge)
    st.line_chart(df_merge)
 

    st.button("Re-run")


if __name__ == '__main__':
    run()
