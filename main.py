import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import read_data_csv, fill_missing_values, train_test_split, scale_data, window_slide
from models.LSTM_autoencoder import LSTM_Autoencoder

def train(params):
    # get hyperparameters
    batch_size = params["batch_size"]
    lr = params["lr"]
    epochs = params["epochs"]

    # read and preprocess data
    df = read_data_csv("data1.csv")
    df = fill_missing_values(df)
    X_train, X_test, y_train, y_test = train_test_split(df)
    X_train, X_test = scale_data(X_train, X_test, list_scale_features=['CO2', 'PM10' , 'PM25' , 'NO', 'NO2', 'O3', 'SO2'])

    # create data loaders for autoencoder
    X_train_final,y_train_final=window_slide(X_train_scaled.values,y_train)
    X_test_final,y_test_final=window_slide(X_test_scaled.values,y_test)
    onehot_labels_train = tf.keras.utils.to_categorical(y_train_final, num_classes=4)
    onehot_labels_test = tf.keras.utils.to_categorical(y_test_final, num_classes=4)

    # define model
    input_shape = X_train.shape[1:]
    model = LSTM_Autoencoder(input_shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    # define criterion
    criterion = nn.MSELoss()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    best_loss = float('inf')
    training_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward pass
            loss.backward()
            optimizer.step()

            # update running variables
            running_loss += loss.item()

        running_loss /= len(train_loader)
        training_loss.append(running_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss:.4f}")

        # save model
        torch.save(model.state_dict(), f'logs/checkpoints/autoencoder_checkpoint_{epoch + 1}.pt')

        if best_loss > running_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), 'logs/checkpoints/best_autoencoder_checkpoint.pt')

    # plot loss
    training_loss = np.array(training_loss)
    plt.plot(training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt
