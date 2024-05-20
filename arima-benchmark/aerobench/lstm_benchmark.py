# ================================================================ #
#                       LSTM Neural Networks                       #
# ================================================================ #
import torch
import torch.nn as nn
from run_GCAS import main_run_gcas
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

# Hyper parameters
import tqdm

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return row['features'], row['label']

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    print(X)
    print(y)
    return torch.tensor(X), torch.tensor(y)
    # return array of lists

batch_size = 10
num_epochs = 10
learning_rate = 0.1

input_dim = 3
hidden_dim = 10
sequence_dim = 28
layer_dim = 1
output_dim = 10

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================ #
#                        Data Loading Process                      #
# ================================================================ #

# Dataset
df = main_run_gcas()
n_steps = 3 # lookback 3 time steps
X, y = split_sequence(df['alt'].values.astype('float32'), n_steps)
df_data_split = pd.DataFrame(columns=['features', 'label'])
df_data_split['features'] = X.tolist()
df_data_split['label'] = y.tolist()

train_size = int(0.7*len(df_data_split)) # Split train and test
train = df_data_split.iloc[0:train_size]
test = df_data_split.iloc[train_size:]

train_dataset = PandasDataset(pd.DataFrame(train))
test_dataset = PandasDataset(pd.DataFrame(test))

# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
)


# ================================================================ #
#                       Create Model Class                         #
# ================================================================ #

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, self.hidden_dim).to(device)
        #c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
    

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])
        
        # out.size() --> 100, 10
        return out


model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Loss function
loss_fn = nn.MSELoss(reduction="mean")

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ================================================================ #
#                           Train and Test                         #
# ================================================================ #

# Train the model
iter = 0
print('TRAINING STARTED.\n')
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = torch.stack(x_batch, dim=1)
        print(x_batch.size())
        x_batch = x_batch.type(torch.FloatTensor)
        outputs = model(x_batch)
        y_batch = y_batch.type(torch.FloatTensor)
        loss = loss_fn(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 10 == 0:
            # Calculate Loss
            print(f'Epoch: {epoch + 1}/{num_epochs}\t Iteration: {iter}\t Loss: {loss.item():.2f}')

