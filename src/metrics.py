def _unique_labels(y_true, y_pred):
    labels = []
    for y in y_true + y_pred:
        if y not in labels:
            labels.append(y)
    labels.sort()
    return labels


def confusion_matrix(y_true, y_pred, labels=None):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = _unique_labels(y_true, y_pred)

    label_to_index = {}
    for idx, label in enumerate(labels):
        label_to_index[label] = idx

    n = len(labels)
    cm = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        cm.append(row)

    for true, pred in zip(y_true, y_pred):
        if true not in label_to_index or pred not in label_to_index:
            raise ValueError("y_true and y_pred contain labels outside the provided labels list")
        i = label_to_index[true]
        j = label_to_index[pred]
        cm[i][j] += 1

    return cm


def accuracy_from_confusion_matrix(cm):
    if not cm:
        return 0.0

    total = 0
    correct = 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            total += cm[i][j]
            if i == j:
                correct += cm[i][j]

    if total > 0:
        return correct / total
    else:
        return 0.0


def precision_recall_f1_from_confusion_matrix(cm):
    n = len(cm)
    precision = []
    recall = []
    f1_score = []

    for i in range(n):
        precision.append(0.0)
        recall.append(0.0)
        f1_score.append(0.0)

    for i in range(n):
        tp = cm[i][i]
        predicted_positive = 0
        actual_positive = 0

        for j in range(n):
            predicted_positive += cm[j][i]  # column sum
            actual_positive += cm[i][j]     # row sum

        if predicted_positive > 0:
            precision[i] = tp / predicted_positive
        else:
            precision[i] = 0.0

        if actual_positive > 0:
            recall[i] = tp / actual_positive
        else:
            recall[i] = 0.0

        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1_score[i] = 0.0

    return precision, recall, f1_score


def accuracy(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    return accuracy_from_confusion_matrix(cm)


def precision(y_true, y_pred, labels=None, average='macro'):
    cm = confusion_matrix(y_true, y_pred, labels)
    precision_vals, _, _ = precision_recall_f1_from_confusion_matrix(cm)

    if average == 'macro':
        if len(precision_vals) > 0:
            return sum(precision_vals) / len(precision_vals)
        else:
            return 0.0
    elif average == 'micro':
        tp_sum = 0
        predicted_sum = 0
        n = len(cm)
        for i in range(n):
            tp_sum += cm[i][i]
            for j in range(n):
                predicted_sum += cm[j][i]
        if predicted_sum > 0:
            return tp_sum / predicted_sum
        else:
            return 0.0
    else:
        raise ValueError("average must be either 'macro' or 'micro'")


def recall(y_true, y_pred, labels=None, average='macro'):
    cm = confusion_matrix(y_true, y_pred, labels)
    _, recall_vals, _ = precision_recall_f1_from_confusion_matrix(cm)

    if average == 'macro':
        if len(recall_vals) > 0:
            return sum(recall_vals) / len(recall_vals)
        else:
            return 0.0
    elif average == 'micro':
        tp_sum = 0
        actual_sum = 0
        n = len(cm)
        for i in range(n):
            tp_sum += cm[i][i]
            for j in range(n):
                actual_sum += cm[i][j]
        if actual_sum > 0:
            return tp_sum / actual_sum
        else:
            return 0.0
    else:
        raise ValueError("average must be either 'macro' or 'micro'")


def f1_score(y_true, y_pred, labels=None, average='macro'):
    cm = confusion_matrix(y_true, y_pred, labels)
    _, _, f1_vals = precision_recall_f1_from_confusion_matrix(cm)

    if average == 'macro':
        if len(f1_vals) > 0:
            return sum(f1_vals) / len(f1_vals)
        else:
            return 0.0
    elif average == 'micro':
        acc = accuracy_from_confusion_matrix(cm)
        return acc
    else:
        raise ValueError("average must be either 'macro' or 'micro'")


def classification_report(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    if labels is None:
        labels = _unique_labels(y_true, y_pred)

    precision_vals, recall_vals, f1_vals = precision_recall_f1_from_confusion_matrix(cm)

    report = {}
    for i, label in enumerate(labels):
        report[str(label)] = {
            'precision': precision_vals[i],
            'recall': recall_vals[i],
            'f1-score': f1_vals[i],
            'support': sum(cm[i])
        }

    acc = accuracy_from_confusion_matrix(cm)
    total_support = 0
    for row in cm:
        for val in row:
            total_support += val

    report['accuracy'] = {
        'precision': acc,
        'recall': acc,
        'f1-score': acc,
        'support': total_support
    }

    macro_prec = precision(y_true, y_pred, labels, average='macro')
    macro_rec = recall(y_true, y_pred, labels, average='macro')
    macro_f1 = f1_score(y_true, y_pred, labels, average='macro')

    report['macro avg'] = {
        'precision': macro_prec,
        'recall': macro_rec,
        'f1-score': macro_f1,
        'support': total_support
    }

    micro_prec = precision(y_true, y_pred, labels, average='micro')
    micro_rec = recall(y_true, y_pred, labels, average='micro')
    micro_f1 = f1_score(y_true, y_pred, labels, average='micro')

    report['micro avg'] = {
        'precision': micro_prec,
        'recall': micro_rec,
        'f1-score': micro_f1,
        'support': total_support
    }

    return report