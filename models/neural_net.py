import pandas as pd
import numpy as np
import torch
import torch.nn
from torch.nn import functional
import torch.utils.data as data
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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
        super(MLP, self).init()
        self.fc1 = nn.Linear(80, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

def get_predictions(model, data_loader):
    """Returns the model's predictions on the given dataset"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(torch.float)
            prediction = model(features).argmax(1).item()
            predictions.append(prediction)
    return predictions

def train(model, train_loader, num_epochs, criterion, optimizer, learning_rate):
    """Trains the model and displays the training accuracy over time"""
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch} Results:')
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(torch.float)
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
        train_acc = accuracy_score(train_loader.y, get_predictions(model, train_loader))
        print('\nAccuracy on training: %.3f%%' % (100*train_acc)))
    return

def metrics(y_val, y_val_preds, print_AUC=False):
    """Prints classifier performance metrics on the validation set"""
    accuracy = accuracy_score(y_val, y_val_preds)
    precision = precision_score(y_val, y_val_preds)
    recall = recall_score(y_val, y_val_preds)
    f1 = f1_score(y_val, y_val_preds)
    print('Metrics on the Validation Set:')
    print('------------------------------')
    print(f'Accuracy: {(accuracy * 100):.3f}%')
    print(f'Precision: {(precision * 100):.3f}%')
    print(f'Recall: {(recall * 100):.3f}%')
    print(f'F1-Score: {(f1 * 100):.3f}%')
    if print_AUC:
        prob_positive = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob_positive)
        print(f'AUC: {auc:.3f}')
    return

if '__name__' == __main__:
    # Set up device #
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # Load training set #
    print('Loading training set...')
    train = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'train_final.csv'))
    print('Completed.')
    features = train.drop(columns=['Severity']).values.astype(float)
    labels = train.loc[:, 'Severity'].values.astype(int)
    train_dataset = AccidentDataset(features, labels)
    # Set hyper-parameters #
    batch_size = 1000
    num_epochs = 10
    learning_rate = 0.01
    # Define train_loader, model, loss fn, and optimizer #
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the network #
    print('Model is training...')
    start = time.time()
    train(model, train_loader, num_epochs, criterion, optimizer, learning_rate)
    end = time.time()
    print(f'Training has ended! Time elapsed: {(end - start)/60:.2f} minutes \n')
    # Load validation set #
    print('Loading vaidation set...')
    val = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'val_final.csv'))
    print('Completed.')
    features_val = val.drop(columns=['Severity']).values.astype(float)
    labels_val = val.loc[:, 'Severity'].values.astype(int)
    val_dataset = AccidentDataset(features, labels)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    y_val = val_loader.y
    y_val_preds = get_predictions(model, val_loader)
    metrics(y_val, y_val_preds)    
