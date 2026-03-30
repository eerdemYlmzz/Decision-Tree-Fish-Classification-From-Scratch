import pandas as pd
import numpy as np
from decision_tree import DecisionTree

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data

def split_data(train_data, test_data):
    X_train = train_data.drop("Species", axis=1)
    y_train = train_data["Species"]
    X_test = test_data.drop("Species", axis=1)
    y_test = test_data["Species"]

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_test, y_test

def k_fold(X, y, loss_method, n_splits=10):
    indices = np.arange(X.shape[0])
    np.random.seed(42) 
    np.random.shuffle(indices)
    
    X_data = X[indices]
    y_data = y[indices]

    loss_history = []
    len_data = X.shape[0]

    fold_size = len_data // n_splits
    
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i != n_splits - 1 else len_data 
        
        X_test, y_test = X_data[start:end, :], y_data[start:end, :]
        
        train_mask = np.ones(len_data, dtype=bool)
        
        train_mask[start:end] = False
        X_train, y_train = X_data[train_mask, :], y_data[train_mask, :]

        model = DecisionTree()
        model = model.fit(X_train, y_train)

        preds = model.predict(X_test)

        loss = loss_method(y_test, preds)

        print(f"Loss {i+1}: {loss:.6f}")
        loss_history.append(loss)

    return loss_history