import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd


def read_data_csv(data_path):
    df = pd.read_csv(data_path)
    return df

def fill_missing_values(df):
    df = df.interpolate()  # Sử dụng hàm interpolate của pandas để điền giá trị thiếu
    df = df.fillna(method='ffill')  # Sử dụng forward fill để điền giá trị thiếu
    df = df.fillna(method='bfill')  # Sử dụng backward fill để điền giá trị thiếu
    return df

def train_test_split(df):
    threshold = int(len(df) * 0.8)
    train = df.iloc[:threshold]
    test = df.iloc[threshold:]

    X_train = torch.tensor(train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train['AQI level'].values, dtype=torch.float32)
    X_test = torch.tensor(test.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(test['AQI level'].values, dtype=torch.float32)

    y_train = torch.reshape(y_train, (y_train.shape[0], 1))
    y_test = torch.reshape(y_test, (y_test.shape[0], 1))
    
    return X_train, X_test, y_train, y_test

def scale_data(df_train, df_test, list_scale_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_train = df_train[list_scale_features].values
    scaled_values_train = scaler.fit_transform(values_train)
    df_train[list_scale_features] = torch.tensor(scaled_values_train, dtype=torch.float32)

    values_test = df_test[list_scale_features].values
    scaled_values_test = scaler.transform(values_test)
    df_test[list_scale_features] = torch.tensor(scaled_values_test, dtype=torch.float32)

    return df_train, df_test

def window_slide(train, label):
    window_size = 10
    X = []
    Y = []
    for i in range(window_size, len(train)):
        X.append(train[i-window_size:i, :])
        Y.append(label[i, :])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
