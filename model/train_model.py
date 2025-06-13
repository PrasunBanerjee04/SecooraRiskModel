import pandas as pd
import requests as r
from datetime import *     
import numpy as np
import os
import sys
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dateutil.parser import isoparse
from transformers import TimesFmModelForPrediction, Trainer, TrainingArguments

class TimeSeriesDataset:
    def __init__(self, series: np.ndarray, context_len: int, horizon_len: int, freq: int):
        self.series = series
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.freq = freq

    def get_latest_context_and_times(self):
        context = self.series[-self.context_len:]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        past_times = torch.arange(self.context_len).unsqueeze(0)
        future_times = torch.arange(self.context_len, self.context_len + self.horizon_len).unsqueeze(0)
        return context_tensor, past_times, future_times
    

if __name__ == "__main__":

    df_1 = pd.read_csv("../data/historical_data_2023.csv", parse_dates=["time"])
    df_1 = df_1.sort_values("time").reset_index(drop=True)
    df_2 = pd.read_csv("../data/historical_data_2024.csv", parse_dates=["time"])
    df_2 = df_2.sort_values("time").reset_index(drop=True)
    df_3 = pd.read_csv("../data/historical_data_2025.csv", parse_dates=["time"])
    df_3 = df_3.sort_values("time").reset_index(drop=True)

    data = pd.concat([df_1, df_2, df_3], ignore_index=True)
    series = data["result"].to_numpy(dtype="float32")
    context_len = 10000
    horizon_len = 100
    freq = 0  

    context_vals = torch.tensor(series[-context_len:], dtype=torch.float32).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch")
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(
            past_values=context_vals.to(device),
            freq=torch.tensor([freq], dtype=torch.long).to(device),
        )
        forecast = output.mean_predictions[0].cpu().numpy()
        print("Forecasted values:", forecast)

    save_dir = "model/timesfm_pretrained"
    os.makedirs(save_dir, exist_ok=True)

    forecast_path = os.path.join(save_dir, "forecast.pkl")
    with open(forecast_path, "wb") as f:
        pickle.dump(forecast, f)

    print(f"Forecast saved to {forecast_path}")
    model.save_pretrained(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))
    print(f"Model saved to {save_dir}/")