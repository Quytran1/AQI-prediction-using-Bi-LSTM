import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import read_data_csv, fill_missing_values, train_test_split, scale_data, window_slide
from models.LSTM_autoencoder import LSTM_Autoencoder
from config import configuration

def train(params):
    batch_size = params["batch_size"]
    lr = params["lr"]
    epochs = params["epochs"]

    df = read_data_csv("data1.csv")
    df = fill_missing_values(df)
    X_train, X_test, y_train, y_test = train_test_split(df)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    X_train_final, y_train_final = window_slide(torch.tensor(X_train_scaled), torch.tensor(y_train))
    X_test_final, y_test_final = window_slide(torch.tensor(X_test_scaled), torch.tensor(y_test))

    # Đảm bảo số lớp bằng số unique giá trị của AQI
    num_classes = len(torch.unique(y_train_final))
    input_shape = X_train_final.shape[1:]

    train_dataset = TensorDataset(X_train_final, y_train_final)
    model = LSTM_Autoencoder(input_shape, num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    training_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets.squeeze(1).long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(train_loader)
        training_loss.append(running_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss:.4f}")

        torch.save(model.state_dict(), f'logs/checkpoints/autoencoder_checkpoint_{epoch + 1}.pt')

        if best_loss > running_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), 'logs/checkpoints/best_autoencoder_checkpoint.pt')

    training_loss = np.array(training_loss)
    plt.plot(training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    train(params=configuration)
