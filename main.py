from src.data_loader import load_and_split, k_fold
from src.decision_tree import DecisionTree
from src.metrics import (confusion_matrix, accuracy, precision, recall, f1_score, 
                         classification_report)
import yaml
import numpy as np

def tune_hyperparameters(X, y, depths, leaves, metric_func):
    best_score = -1
    best_params = {}
    results = []

    print("\n" + "="*50)
    print(f"{'Max Depth':<12} | {'Min Leaf':<10} | {'Mean Accuracy'}")
    print("-" * 50)

    for d in depths:
        for l in leaves:
            scores = k_fold(X, y, metric_func=metric_func, n_splits=10, 
                            max_depth=d, min_samples_leaf=l)
            
            mean_score = sum(scores) / len(scores)
            results.append((d, l, mean_score))
            
            print(f"{d:<12} | {l:<10} | %{mean_score*100:.2f}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'max_depth': d, 'min_samples_leaf': l}

    print("="*50)
    print(f"BEST RESULTS: %{best_score*100:.2f}")
    print(f"BEST PARAMETERS: {best_params}")
    
    return best_params

def print_confusion_matrix(cm, labels):
    print("\n" + " "*15 + "PREDICTED")
    header = "          " + "".join([f"{str(l):>10}" for l in labels])
    print(header)
    print("ACTUAL" + "   " + "-" * (len(header) - 10))
    for i, label in enumerate(labels):
        row = f"{str(label):<10}|" + "".join([f"{val:>10}" for val in cm[i]])
        print(row)

def print_prediction_table(y_true, y_pred, num_samples=10):
    print("\n" + "="*50)
    print(f"{'Sample':<10} | {'Real Species':<15} | {'Guessed':<15}")
    print("-" * 45)
    for i in range(min(num_samples, len(y_true))):
        print(f"{i+1:<10} | {str(y_true[i]):<15} | {str(y_pred[i]):<15}")
    print("="*50)

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file) 

    train_path = config["paths"]["train_data"]
    test_path = config["paths"]["test_data"]

    X_train, y_train, X_test, y_test = load_and_split(train_path, test_path)

    depth_list = [3, 4, 5, 6, 8]
    leaf_list = [2, 5, 8, 10]

    print("\nHYPERPARAMETER OPTIMIZATION HAS STARTED...")
    best_config = tune_hyperparameters(X_train, y_train, depth_list, leaf_list, accuracy)

    print("\nBEST PARAMETERS ARE TESTED...")
    final_model = DecisionTree(
        max_depth=best_config['max_depth'], 
        min_samples_leaf=best_config['min_samples_leaf']
    )
    final_model.fit(X_train, y_train)

    print("\n10-FOLD CROSS VALIDATION ON TEST DATA")
    acc_scores = k_fold(X_train, y_train, n_splits=10, max_depth=best_config["max_depth"], 
                        min_samples_leaf=best_config["min_samples_leaf"], metric_func=accuracy)
    
    mean_acc = sum(acc_scores) / len(acc_scores)
    print(f"\n10-Fold Average Accuracy: %{mean_acc*100:.2f}")

    print("\nFINAL EVALUATION")
    y_preds = final_model.predict(X_test)
    
    print_prediction_table(y_test, y_preds, num_samples=15)

    labels = sorted(list(set(y_test))) 
    cm = confusion_matrix(y_test.tolist(), y_preds, labels=labels)
    
    print("\nCONFUSION MATRIX")
    print_confusion_matrix(cm, labels)

    print("\nPERFORMANCE METRICS")
    test_acc = accuracy(y_test.tolist(), y_preds)
    test_prec = precision(y_test.tolist(), y_preds, average='macro')
    test_rec = recall(y_test.tolist(), y_preds, average='macro')
    test_f1 = f1_score(y_test.tolist(), y_preds, average='macro')

    print(f"Test Accuracy  : %{test_acc*100:.2f}")
    print(f"Test Precision : %{test_prec*100:.2f}")
    print(f"Test Recall    : %{test_rec*100:.2f}")
    print(f"Test F1-Score  : %{test_f1*100:.2f}")

if __name__ == "__main__":
    main()