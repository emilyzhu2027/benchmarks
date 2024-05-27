import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data
np.random.seed(42)
time_series_length = 1000
time = np.arange(0, time_series_length)
data = np.sin(0.02 * time) + 0.5 * np.random.normal(size=time_series_length)

# Plot the synthetic data
plt.figure(figsize=(12, 6))
plt.plot(time, data)
plt.title('Synthetic Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step)]
        X.append(a)
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data)

# Split the data into train and test sets
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

# Create dataset for training
time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Initialize the model, define the loss function and the optimizer
model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
for epoch in range(epochs):
    for i in range(len(X_train)):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        optimizer.zero_grad()
        y_pred = model(X_train[i].unsqueeze(0))
        single_loss = loss_function(y_pred, y_train[i].unsqueeze(0))
        single_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} loss: {single_loss.item()}')

# Make predictions
model.eval()
with torch.no_grad():
    train_predict = model(X_train).numpy()
    test_predict = model(X_test).numpy()

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.unsqueeze(1).numpy())
y_test = scaler.inverse_transform(y_test.unsqueeze(1).numpy())

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(time, scaler.inverse_transform(data), label='Original Data')
plt.plot(time[time_step:len(train_predict)+time_step], train_predict, label='Train Prediction')
plt.plot(time[len(train_predict)+(time_step*2)+1:len(data)-1], test_predict, label='Test Prediction')
plt.legend()
plt.title('Original Data vs Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
