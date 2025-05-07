import pandas as pd
import requests as r
from datetime import datetime, timedelta 
import numpy as np  
from concurrent.futures import ThreadPoolExecutor
from dateutil.parser import isoparse
from tqdm.auto import tqdm

class DataLoader:
    
    @classmethod
    def extract_data(cls, start_date: str, end_date: str) -> pd.DataFrame:
        base_url = "https://api.sealevelsensors.org/v1.0/Datastreams(262)/Observations"
        url = (
            f"{base_url}"
            f"?$orderby=phenomenonTime%20desc"
            f"&$filter=phenomenonTime%20ge%20{start_date}"
            f"%20and%20phenomenonTime%20le%20{end_date}"
        )

        all_results = {"time": [], "result": []}
        while url:
            response = r.get(url)
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}")
            data = response.json()
            for obs in data.get("value", []):
                all_results["time"].append(obs["phenomenonTime"])
                all_results["result"].append(obs["result"])
            url = data.get("@iot.nextLink")

        df = pd.DataFrame(all_results)
        df["time"] = pd.to_datetime(df["time"], utc=True, infer_datetime_format=True, errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.sort_values("time", ascending=False).reset_index(drop=True)
        return df

    @classmethod
    def extract_data_concurrent(cls, start_date: str, end_date: str, num_workers: int = 8) -> pd.DataFrame:
        dt_start = isoparse(start_date)
        dt_end = isoparse(end_date)

        total_seconds = (dt_end - dt_start).total_seconds()
        chunk = total_seconds / num_workers

        fmt = "%Y-%m-%dT%H:%M:%SZ"
        ranges = []
        for i in range(num_workers):
            sub_s = dt_start + timedelta(seconds=chunk * i)
            sub_e = (dt_start + timedelta(seconds=chunk * (i + 1))) if i < num_workers - 1 else dt_end
            ranges.append((sub_s.strftime(fmt), sub_e.strftime(fmt)))

        def fetch(rng):
            return cls.extract_data(rng[0], rng[1])

        dfs = []
        with ThreadPoolExecutor(max_workers=num_workers) as exe:
            for df_chunk in tqdm(exe.map(fetch, ranges), total=len(ranges), desc="Fetching data"):
                dfs.append(df_chunk)

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("time", ascending=False).reset_index(drop=True)
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

        
if __name__ == "__main__":

    start_date = "2024-01-01T00:00:00Z"
    end_date   = "2024-12-31T00:00:00Z"

    df = DataLoader.extract_data_concurrent(
        start_date=start_date,
        end_date=end_date,
        num_workers=10
    )

    #take pandas dataframe --> csv file 
    DataLoader.load_data_to_csv(df, "data/historical_data_2024.csv")
    