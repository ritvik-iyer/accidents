import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.utils import shuffle
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

class AccidentTestset(data.Dataset):
    def __init__(self, features):
        self.X = features
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        X_idx = self.X[idx, :]
        return X_idx

class MLP(nn.Module):
    def __init__(self):
        """Define the MLP architecture"""
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 800)
        self.fc5 = nn.Linear(800, 512)
        self.fc6 = nn.Linear(512, 50)
        self.fc7 = nn.Linear(50, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(800)
        self.batchnorm4 = nn.BatchNorm1d(50)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.batchnorm3(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.fc5(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc6(x))
        x = self.batchnorm4(x)
        x = self.dropout1(x)
        x = self.fc7(x)
        return x
    def validate(self, data_loader, device=torch.device('cpu')):
        self.eval()
        correct = 0
        with torch.no_grad():
            for features, labels in data_loader:
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
            # Perform backpropagation
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

def load_model(file_name, device):
    """Loads PyTorch model with specified model name"""
    model_path = os.path.join(os.path.join(os.getcwd(), 'saved_models'), file_name)
    assert os.path.exists(model_path), f"{file_name} does not exist!"
    try:
        model = MLP().to(device)
        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        raise ValueError('Unable to load model')

def inference(model, test_loader, test_ids, model_name):
    """Runs inference on the test set and writes results to file"""
    IDs = test_ids
    model.eval()
    test_set_preds = []
    with torch.no_grad():
        for features in test_loader:
            features = features.to(torch.float)
            features = features.to(device)
            outputs = torch.argmax(model(features), dim=1).tolist()
            test_set_preds.extend(outputs)
    predictions = pd.DataFrame({'ID':IDs,
                                'Severity':test_set_preds})
    pred_file_name = f'{model_name}.csv'
    full_file_path = os.path.join(os.path.join(os.path.dirname(os.getcwd()), 'predictions'), pred_file_name)
    predictions.to_csv(full_file_path, index=False)
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
    learning_rate = 0.05
    # Define datasets #
    print('Loading training set...')
    train = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'train_sampled.csv'))
    print('Completed.')
    features_train = train.drop(columns=['Severity']).loc[:, 'Temperature':'Blocked'].values.astype(float)
    labels_train = train.loc[:, 'Severity'].values.astype(int)
    train_dataset = AccidentDataset(features_train, labels_train)
    train_len = features_train.shape[0]
    print('Loading validation set...')
    val = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'val_final.csv'))
    print('Completed.')
    features_val = val.drop(columns=['Severity']).loc[:, 'Temperature':'Blocked'].values.astype(float)
    labels_val = val.loc[:, 'Severity'].values.astype(int)
    val_dataset = AccidentDataset(features_val, labels_val)
    val_len = features_val.shape[0]
    # Convert data sets to dataloaders #
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
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
    file_name = '5HL_Batchnorm_005SGD_Numerical_10Epochs_CELoss'
    torch.save(model.state_dict(), os.path.join(os.path.join(os.getcwd(), 'saved_models'), file_name))
    # Load the test set #
    print('Loading test set...')
    test = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'test_final.csv'))
    print('Completed.')
    # Create dataloader for test set #
    IDs = test['ID']
    features_test = test.drop(columns=['ID']).loc[:, 'Temperature':'Blocked'].values.astype(float)
    test_dataset = AccidentTestset(features_test)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    # Load the model #
    print('Loading the model...')
    model = load_model(file_name, device)
    print('Completed.')
    # Perform inference on test set and write results to file #
    inference(model, test_loader, IDs, file_name)
