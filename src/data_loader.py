import pandas as pd
import numpy as np
from src.decision_tree import DecisionTree

def load_and_split(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop("Species", axis=1)
    y_train = train_data["Species"]
    X_test = test_data.drop("Species", axis=1)
    y_test = test_data["Species"]

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_test, y_test

def k_fold(X, y, metric_func, n_splits=10, max_depth=8, min_samples_leaf=2):    
    indices = np.arange(X.shape[0])
    np.random.seed(42) 
    np.random.shuffle(indices)
    
    X_data = X[indices]
    y_data = y[indices]

    results_history = []
    len_data = X.shape[0]
    fold_size = len_data // n_splits
    
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i != n_splits - 1 else len_data 
        
        X_test, y_test = X_data[start:end, :], y_data[start:end]
        
        train_mask = np.ones(len_data, dtype=bool)
        train_mask[start:end] = False
        X_train, y_train = X_data[train_mask, :], y_data[train_mask]

        model = DecisionTree(min_samples_leaf=min_samples_leaf, max_depth=max_depth)        
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        y_true_list = y_test.tolist()
        y_pred_list = list(preds)

        score = metric_func(y_true_list, y_pred_list)
        results_history.append(score)
        
        print(f"Fold {i+1} Score: {score:.4f}")

    return results_history