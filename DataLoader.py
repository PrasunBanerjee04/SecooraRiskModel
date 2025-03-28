import pandas as pd
import requests as r
from datetime import *   
import numpy as np     

class DataLoader:

    def __init__(self, url):
        self.api_url = url
    
    @classmethod
    def extract_data(cls, numOfDays):
        original_link = "https://api.sealevelsensors.org/v1.0/Datastreams(262)/Observations?$skip=0&$orderby=%40iot.id+desc&"
        now = datetime.now(timezone.utc)
        yest = (now - timedelta(days = numOfDays)).strftime('%Y-%m-%d')
        formatted_time = now.strftime("T%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        now = now.strftime('%Y-%m-%d')
        link = original_link + "$filter=phenomenonTime%20ge%20" + yest + formatted_time + "%20and%20phenomenonTime%20le%20" + now + formatted_time
        metadata = r.get(link)
        data = metadata.json()
        adict = {"resultTime":[], "reading":[]}
        while "@iot.nextLink" in data:
            for dictionary in data["value"]:
                rTime = dictionary["resultTime"]
                result = dictionary["result"]
                adict["resultTime"].append(rTime)
                adict["reading"].append(result)
            link = data["@iot.nextLink"]
            metadata = r.get(link)
            data = metadata.json()
        df = pd.DataFrame(adict)
        return df

    
    @classmethod
    def transform_data(cls, df):
        #take two column data and integrate advanced features
        df["time"] = pd.to_datetime(df["time"])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["day_of_week"] = df["time"].dt.dayofweek
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["second"] = df["time"].dt.second

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["lag_1"] = df["result"].shift(1)
        df["lag_2"] = df["result"].shift(2)
        df["lag_3"] = df["result"].shift(3)

        df["rolling_mean_3"] = df["result"].rolling(window=3).mean()
        df["rolling_std_3"] = df["result"].rolling(window=3).std()
        
        return df
    
    @classmethod
    def order_df(cls, df):
        #order the columns of the dataframe
        columns = ["year", 
                    "month", 
                    "day", 
                    "day_of_week", 
                    "hour", 
                    "minute", 
                    "second", 
                    "hour_sin", 
                    "hour_cos", 
                    "day_of_week_sin", 
                    "day_of_week_cos", 
                    "month_sin", 
                    "month_cos", 
                    "lag_1",
                    "lag_2", 
                    "lag_3", 
                    "rolling_mean_3",
                    "rolling_std_3",
                    "result"
                    ]

        df = df[columns]
        return df

    
    @classmethod
    def load_data_to_csv(cls, df, path):
        #take pandas dataframe --> csv file 
        df.to_csv(path, index=False)
        
        
    
    @classmethod
    def load_data_to_parquet(cls, df, path):
        #take pandas dataframe --> parquet file 
        df.to_parquet(path, index=False)  # 'path' should be a string with the full file path
        
