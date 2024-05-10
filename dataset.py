import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd

def read_data_csv(data_path):
    df = pd.read_csv(data_path)
    return df

def fill_missing_values(df):
    df = df.interpolate()
    df = df.ffill()
    df = df.bfill()
    return df

def train_test_split(df):
    threshold = int(len(df) * 0.8)
    train = df.iloc[:threshold]
    test = df.iloc[threshold:]

    X_train = torch.tensor(train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train['AQI'].values, dtype=torch.float32)
    X_test = torch.tensor(test.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(test['AQI'].values, dtype=torch.float32)

    y_train = torch.reshape(y_train, (y_train.shape[0], 1))
    y_test = torch.reshape(y_test, (y_test.shape[0], 1))
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    X_train_scaled = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    return X_train_scaled, X_test_scaled

def window_slide(train, label):
    window_size = 10
    X = []
    Y = []
    for i in range(window_size, len(train)):
        X.append(train[i-window_size:i, :])
        Y.append(label[i, :])
    return torch.stack(X), torch.stack(Y)
