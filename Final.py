# In this project we'll be using the pytorch library to predict traffic volume on a given interstate highway in minnesota. The dataset is available on the UCI machine learning repository. 
# There is 4 main steps in this project:
# 1. Prepare the data for modelling,
# 2. Creating the NN model,
# 3. Training the model,
# 4. Evaluating the model.


#################### 1. PREPARING THE DATA ####################


# To model time-series data, we need to generate sequences of past values as inputs and predict the next
# value as the target. One way to do this is by writing a function and passing in the available data.
# These need to be converted to pytorsh tensors and loaded with DataLoader.
# 1. Creating sequences with a function
# 2. Converting to tensors
# 3. Loading data

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics


# Creating the sequences 

# read the traffic data from the csv training and test files

train_scaled_df = pd.read_csv('E:/PROJECTS/Traffic Flow/train_scaled.csv')
test_scaled_df = pd.read_csv("E:/PROJECTS/Traffic Flow/test_scaled.csv")

# convert the dataframes to numpy arrays

train_scaled = train_scaled_df.to_numpy()
test_scaled = test_scaled_df.to_numpy()

# define a function to create sequences of data, the function will take the datasets, sequence length, and target column index as inputs
# and return the sequences and targets as numpy arrays

# Note: in our dataset, the target column is the last column (Traffic Volume) but for flexibility
# we'll pass the target column index as an argument 

def create_sequences(Data, seq_length, target_col):
    xs = []
    ys = []
    for i in range(len(Data)-seq_length):
        x = Data[i:(i+seq_length)]
        y = Data[i+seq_length, target_col]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

# so this function generates (input, target) pairs for the model to train on
# the input is a sequence of len(seq_length) and the target is the next value in the sequence, the last value, traffic volume.

# we'll apply this function on the test and train datasets

X_train, Y_train = create_sequences(train_scaled, 5, 65)

X_test, Y_test = create_sequences(test_scaled, 5, 65)

# convert the numpy arrays to pytorch tensors

X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()

X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()

# now we'll load the data using DataLoader

train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


#################### 2. CREATING THE MODEL ####################

# An appropriate choice for the model is a LSTM (Long Short Term Memory) network. This is because recurrent neural networks
# are good for temporal dependancies in sequential data. LSTM is a type of RNN that can remember past values and use them to predict future values. 
# in LSTM a cell takes three inputs, because there's two hidden states, long term and short term
# and produces an output with two new hidden states. This is useful for time-series data because it can
# remember past values and use them to predict future values. while also forgetting irrelevant information. 

# there are main steps in creating the model:
# 1. choosing the right neural network (LSTM)
# 2. defining the model
# 3. choosing an activation function

# 6. defining the forward pass

# define input size to the model

NumberofFeatures = X_train.shape[2]

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size = NumberofFeatures,
            hidden_size = 64, 
            num_layers = 2,
            batch_first = True, 
            
            # incorporate dropout
            
            dropout = 0.3
        )
        self.dropout = nn.Dropout(p = 0.3) # Dropout before the fully connected layer
        self.fc = nn.Linear(64, 1) # Fully connected layer to map LSTM output to single prediction
    
    def forward(self, x):
        
        # initialize hidden states
        
        h_0 = torch.zeros(2, x.size(0), 64)
        c_0 = torch.zeros(2, x.size(0), 64)
        
        # pass input through LSTM layer
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # apply dropout before the fully connected layer
        
        out = self.dropout(out[:, -1, :])
        
        # pass the output of the LSTM layer from the last time step to the fully connected layer
        
        out = self.fc(out)
        
        out = F.relu(out)
        
        return out
    
    
# Initializing the model
    
traffic_model = Net(NumberofFeatures)

# defining the loss function and optimizer
    
criterion = nn.MSELoss()
optimizer = optim.Adam(traffic_model.parameters(), lr=0.0001)   
    
num_epochs = 20
    
for epochs in range(num_epochs):
    
    total_loss = 0  # to keep track of the loss
    num_batches = len(train_loader) 
    
    for seqs, labels in train_loader:
        
        seq = seqs.view(seqs.size(0), 5, NumberofFeatures)
        
        optimizer.zero_grad()
        
        outputs = traffic_model(seqs)
        loss = criterion(outputs.squeeze(), labels)
        
        
        
        loss.backward() 
        optimizer.step()
        
        total_loss += loss.item() # accumulate the loss
    
    avg_loss = total_loss / num_batches
    print("Epoch: {}, Loss: {}".format(epochs, avg_loss))
        
        
        
        
        
#################### 4. EVALUATING THE MODEL ####################

# To evaluate the model, we'll use the test data to make predictions and compare them to the actual values.
# We'll use the mean squared error as the evaluation metric.


# set model to evaluation

traffic_model.eval()

# disable gradient computation, this reduces memory usage, speed up computation, and prevents accidental model updates
        
with torch.no_grad():
    total_mse = 0
    num_batches = len(test_loader)
    
    # loop through the test dataset
    
    for seqs, labels in test_loader:
        
        seqs = seqs.view(seqs.size(0), 5, NumberofFeatures) # ensure correct shape
        
        
        predictions = traffic_model(seqs) # Forward pass
        
        mse = F.mse_loss(predictions.squeeze(), labels, reduction='sum') # find MSE per batch
        
        total_mse += mse.item() # accumulated mse
        

# compute overall mse

avg_mse = total_mse /len(test_data)
        
print("average MSE: ", avg_mse)

