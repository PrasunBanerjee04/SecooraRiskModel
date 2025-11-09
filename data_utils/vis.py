## -- VISUALIZATION METHODS -- 
import pandas as pd
import requests as r
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

def visualize_time_series(df, X = None, title = "Time Series Plot"):
    if X == None:
        df.plot(figsize=(12,6), title=title)
    else:
        df.plot(x=X, figsize=(12,6), title=title)
    
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    plt.show()

def visualize_histogram(series, bins=500, title="Histogram"):
    series.plot(kind='hist', bins=bins, title=title)
    plt.show()

def plot_outliers(df, threshold, column):
    mask = df[column].abs() > threshold
    ax = df.loc[mask, column].plot(figsize=(12, 6), style='.')
    ax.set_xlabel(df.index.name or "index")
    ax.set_ylabel("result")
    return ax

def visualize_split(df_train, df_test, split_date):
    plt.figure(figsize=(12,6))
    plt.plot(df_train['ds'], df_train['y'], label='Training Data', color='darkblue', linewidth=1.5)
    plt.plot(df_test['ds'], df_test['y'], label='Test Data', color='crimson', linewidth=1.5)
    plt.axvline(x=split_date, color='black', linestyle='--', linewidth=2, label='Split Date')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_forecast(df_train, forecasts, split_date):
    plt.figure(figsize=(12,6))
    plt.plot(df_train['ds'], df_train['y'], label='Training Data', color='darkblue', linewidth=1.5)
    plt.plot(forecasts['ds'], forecasts['yhat'], label='Forecasts', color='lightblue', linewidth=1.5)
    plt.axvline(x=split_date, color='black', linestyle='--', linewidth=2, label='Split Date')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()