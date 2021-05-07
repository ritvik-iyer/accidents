import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pickle
import os
import time

def train_model(train, file_name):
    """Trains a Random Forest classification model and writes it to a file"""
    pass

def metrics(val, model, print_AUC=False):
    """Prints classifier performance metrics on the validation set"""
    X_val = val.drop(columns=['Severity']).values.astype(float)
    y_val = val.loc[:, 'Severity'].values.astype(int)
    y_val_preds = model.predict(X_val)
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

def inference(test, model, model_name, write_to_file=True):
    """Runs inference on the test set and optionally writes results to file"""
    IDs = test['ID']
    X_test = test.drop(columns=['ID'])
    y_test_preds = model.predict(X_test)
    predictions = pd.DataFrame({'ID':IDs,
                                'Severity':y_test_preds})
    if write_to_file:
        pred_file_name = f'{model_name}.csv'
        full_file_path = os.path.join(os.path.join(os.path.dirname(os.getcwd()), 'predictions'), pred_file_name)
        predictions.to_csv(full_file_path, index=False)

def load_model(file_name):
    """Loads model with specified file name"""
    model_path = os.path.join(os.path.join(os.getcwd(), 'saved_models'), file_name)
    assert os.path.exists(model_path), f"{file_name} does not exist!"
    try:
        with open(model_path, 'rb') as handle:
            model = pickle.load(handle)
        return model
    except Exception as e:
        raise ValueError('Unable to load model')

if __name__ == '__main__':
    model_name = 'random_forest'
    file_name = f'{model_name}.sav'
    if not os.path.exists(os.path.join('saved_models', file_name)):
        train = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'train_final.csv'))
        train_model(train, file_name)
    model = load_model(file_name)
    val = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'val_final.csv'))
    metrics(val, model, print_AUC=True)
    test = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'test_final.csv'))
    inference(test, model, model_name)
