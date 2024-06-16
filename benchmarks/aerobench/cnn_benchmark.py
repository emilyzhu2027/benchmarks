import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from run_GCAS import main_run_gcas

# Define a function to create sequences for training
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step)]
        X.append(a)
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

seq_length = 10

# Define the CNN model in PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (seq_length - 2) // 2, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

results_df = pd.DataFrame(columns=['trial_num', 'test_rmse'])
for i in range(20):
    df = main_run_gcas()
    data = df['alt'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(data)

    train_size = int(len(data) * 0.67)
    test_size = len(data) - train_size
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)

    # Create DataLoader for batching
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    # Train the model
    model.train()
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = []
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            y_pred.extend(outputs.squeeze().numpy())
        y_pred = np.array(y_pred)

    test_mse = criterion(torch.tensor(scaler.inverse_transform(y_test)), torch.tensor(scaler.inverse_transform(y_pred.reshape(-1, 1)))).item()
    test_rmse = np.sqrt(test_mse)

    results_df = pd.concat([pd.DataFrame([[i+1, test_rmse]], columns=results_df.columns), results_df], ignore_index=True)

    """
    # Plot the original and predicted time series
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test.numpy(), label='True')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred, label='Predicted')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Forecasting with CNN')
    plt.show()
    """

results_df.to_csv("cnn_benchmark_results.csv")

