import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import os
import time

class AccidentDataset(data.Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        X_idx = self.X[idx]
        y_idx = self.y[idx]
        return X_idx, y_idx

class MLP(nn.Module):
    def __init__(self):
        """Define the MLP architecture"""
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(80, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    def validate(self, val_loader, device=torch.device('cpu')):
        correct = 0
        for features, labels in val_loader:
            features = features.to(torch.float)
            labels = labels.to(torch.long)
            features = features.to(device)
            labels = labels.to(device)
            outputs = torch.argmax(self(features), dim=1)
            correct += int(torch.sum(outputs==labels))
        return correct

def train_network(model, train_loader, val_loader, train_len, val_len,
                  num_epochs, criterion, optimizer, learning_rate, device):
    """Trains the model and displays the training accuracy over time"""
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch + 1} Results:')
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Type conversions
            features = features.to(torch.float)
            labels = labels.to(torch.long)
            # Send batch to device/GPU
            features = features.to(device)
            labels = labels.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            output = model(features)
            # Compute loss
            loss = criterion(output, labels)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            # Print statistics
            if batch_idx % 5000 == 0:
                print("Batch %d/%d, Loss=%.4f" % (batch_idx, len(train_loader), loss.item()))
        #train_acc = accuracy_score(get_labels(train_loader), get_predictions(model, train_loader))
        #print('\nAccuracy on training: %.3f%%' % (100*train_acc))
        print(f'Training Accuracy: {(model.validate(train_loader, device=device)/train_len) * 100:.3f}%')
        print(f'Validation Accuracy: {(model.validate(val_loader, device=device)/val_len) * 100:.3f}%')
    return

if __name__ == '__main__':
    # Set up device #
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Device set to: {torch.cuda.get_device_name()}')
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # Set hyper-parameters #
    batch_size = 100
    num_epochs = 10
    learning_rate = 0.01
    # Define datasets #
    print('Loading training set...')
    train = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'train_final.csv'))
    print('Completed.')
    features_train = train.drop(columns=['Severity']).values.astype(float)
    labels_train = train.loc[:, 'Severity'].values.astype(int)
    train_dataset = AccidentDataset(features_train, labels_train)
    train_len = features_train.shape[0]
    print('Loading validation set...')
    val = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'val_final.csv'))
    print('Completed.')
    features_val = val.drop(columns=['Severity']).values.astype(float)
    labels_val = val.loc[:, 'Severity'].values.astype(int)
    val_dataset = AccidentDataset(features_val, labels_val)
    val_len = features_val.shape[0]
    # Convert data sets to dataloaders #
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # Define model, loss fn, and optimizer #
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Train the network #
    print('Model is training...')
    start = time.time()
    train_network(model, train_loader, val_loader, train_len, val_len,
    num_epochs, criterion, optimizer, learning_rate, device)
    end = time.time()
    print(f'Training has ended! Time elapsed: {(end - start)/60:.2f} minutes \n')
    # Save the model #
    file_name = 'MLP-1HL'
    torch.save(model.state_dict(), os.path.join(os.path.join(os.getcwd(), 'saved_models'), file_name))
