import pandas as pd
import requests as r
from datetime import *     
import numpy as np
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dateutil.parser import isoparse
from transformers import TimesFmModelForPrediction, Trainer, TrainingArguments

class TimeSeriesDataset(Dataset):

    def __init__(self, series: np.ndarray, context_len: int, horizon_len: int, freq: int):

        n = len(series) - context_len - horizon_len + 1
        self.past = torch.empty((n, context_len), dtype=torch.float32)
        self.future = torch.empty((n, horizon_len), dtype=torch.float32)
        self.freqs = torch.full((n,), freq, dtype=torch.long)

        for i in range(n):
            self.past[i] = torch.from_numpy(series[i : i + context_len])
            self.future[i] = torch.from_numpy(series[i + context_len : i + context_len + horizon_len])

    def __len__(self):
        return self.past.size(0)
    
    def __getitem__(self, idx):
        return {
            "past_values": self.past[idx],
            "freq": self.freqs[idx],
            "future_values": self.future[idx]

        }
    

if __name__ == "__main__":

    df_1 = pd.read_csv("../data/historical_data_2023.csv", parse_dates=["time"])
    df_1 = df_1.sort_values("time").reset_index(drop=True)

    df_2 = pd.read_csv("../data/historical_data_2024.csv", parse_dates=["time"])
    df_2 = df_2.sort_values("time").reset_index(drop=True)

    df_3 = pd.read_csv("../data/historical_data_2025.csv", parse_dates=["time"])
    df_3 = df_3.sort_values("time").reset_index(drop=True)

    data = pd.concat([df_1, df_2, df_3], ignore_index=True)
    data = data[0:130254]
    series = data["result"].to_numpy(dtype="float32")

    context_len = 512 
    horizon_len = 128
    freq = 0 

    N = len(series)
    validation_frac = 0.15
    test_frac = 0.15 

    train_end = int(N * (1 - validation_frac - test_frac))
    validation_end = int(N * (1 - test_frac))

    train_series = series[:train_end]
    validation_series = series[train_end - context_len : validation_end]
    test_series = series[validation_end - context_len : ]

    train_ds = TimeSeriesDataset(train_series, context_len, horizon_len, freq)
    validation_ds = TimeSeriesDataset(validation_series, context_len, horizon_len, freq)
    test_ds = TimeSeriesDataset(test_series, context_len, horizon_len, freq)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimesFmModelForPrediction.from_pretrained(
        "google/timesfm-2.0-500m-pytorch", 
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.train()

    training_args = TrainingArguments(
    output_dir="timesfm_finetune",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
    )
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=validation_ds)
    print("Validation loss:", val_metrics["eval_loss"])

    metrics = trainer.evaluate(eval_dataset=test_ds)
    print(metrics)

    trainer.save_model("model/timesfm_finetune")
    torch.save(model.state_dict(), "model/model.pkl")
    print("Model saved to model/model.pkl and model/timesfm_finetune/")