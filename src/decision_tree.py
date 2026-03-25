import math


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples


class DecisionTree:
    def __init__(self, min_samples_leaf=2, max_depth=20):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = None
        self.classes = None
    
    def entropy_calculation(self, labels):
        if len(labels) == 0:
            return 0
        
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        if len(class_counts) == 1:
            return 0
        
        entropy = 0
        total = len(labels)
        
        for count in class_counts.values():
            if count > 0:
                p_i = count / total
                entropy -= p_i * math.log2(p_i)
        
        return entropy
    
    def information_gain(self, parent_labels, left_labels, right_labels):
        if len(parent_labels) == 0:
            return 0
        
        w_left = len(left_labels) / len(parent_labels)
        w_right = len(right_labels) / len(parent_labels)
        
        h_parent = self.entropy_calculation(parent_labels)
        h_left = self.entropy_calculation(left_labels)
        h_right = self.entropy_calculation(right_labels)
        
        gain = h_parent - (w_left * h_left + w_right * h_right)
        
        return gain if gain > 0 else 0
    
    def find_best_split(self, features, labels):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = len(features[0]) if len(features) > 0 else 0
        
        for feature_idx in range(n_features):
            feature_values = []
            for i in range(len(features)):
                feature_values.append(features[i][feature_idx])
            
            unique_values = []
            for val in feature_values:
                if val not in unique_values:
                    unique_values.append(val)
            
            unique_values.sort()
            
            for threshold in unique_values:
                left_labels = []
                right_labels = []
                
                for i in range(len(features)):
                    if features[i][feature_idx] <= threshold:
                        left_labels.append(labels[i])
                    else:
                        right_labels.append(labels[i])
                
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue
                
                gain = self.information_gain(labels, left_labels, right_labels)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, features, labels, depth=0):
        unique_classes = []
        for label in labels:
            if label not in unique_classes:
                unique_classes.append(label)
        
        n_samples = len(labels)
        
        if len(unique_classes) == 1:
            leaf = TreeNode(value=unique_classes[0], samples=n_samples)
            return leaf
        
        if n_samples <= self.min_samples_leaf:
            class_counts = {}
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            majority_class = max(class_counts, key=class_counts.get)
            leaf = TreeNode(value=majority_class, samples=n_samples)
            return leaf
        
        if depth >= self.max_depth:
            class_counts = {}
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            majority_class = max(class_counts, key=class_counts.get)
            leaf = TreeNode(value=majority_class, samples=n_samples)
            return leaf
        
        best_feature, best_threshold, best_gain = self.find_best_split(features, labels)
        
        if best_feature is None or best_gain == 0:
            class_counts = {}
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            majority_class = max(class_counts, key=class_counts.get)
            leaf = TreeNode(value=majority_class, samples=n_samples)
            return leaf
        
        left_features = []
        left_labels = []
        right_features = []
        right_labels = []
        
        for i in range(len(features)):
            if features[i][best_feature] <= best_threshold:
                left_features.append(features[i])
                left_labels.append(labels[i])
            else:
                right_features.append(features[i])
                right_labels.append(labels[i])
        
        left_subtree = self.build_tree(left_features, left_labels, depth + 1)
        right_subtree = self.build_tree(right_features, right_labels, depth + 1)
        
        node = TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            samples=n_samples
        )
        
        return node
    
    def predict_sample(self, node, features_row):
        if node.value is not None:
            return node.value
        
        if features_row[node.feature] <= node.threshold:
            return self.predict_sample(node.left, features_row)
        else:
            return self.predict_sample(node.right, features_row)
    
    def fit(self, X, y):
        if hasattr(X, 'values'):
            self.feature_names = list(X.columns)
            features = [list(row) for row in X.values]
        else:
            self.feature_names = None
            features = X
        
        if hasattr(y, 'values'):
            labels = list(y.values)
        else:
            labels = y
        
        self.classes = []
        for label in labels:
            if label not in self.classes:
                self.classes.append(label)
        
        self.tree = self.build_tree(features, labels, depth=0)
        
        return self
    
    def predict(self, X):
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        if hasattr(X, 'values'):
            features = [list(row) for row in X.values]
        else:
            features = X
        
        predictions = []
        for features_row in features:
            prediction = self.predict_sample(self.tree, features_row)
            predictions.append(prediction)
        
        return predictions
    
    def get_tree_depth(self, node=None):
        if node is None:
            node = self.tree
        
        if node is None:
            return 0
        
        if node.value is not None:
            return 0
        
        left_depth = self.get_tree_depth(node.left)
        right_depth = self.get_tree_depth(node.right)
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node=None):
        if node is None:
            node = self.tree
        
        if node is None:
            return 0
        
        count = 1
        if node.left is not None:
            count += self.count_nodes(node.left)
        if node.right is not None:
            count += self.count_nodes(node.right)
        
        return count


def predict_batch(tree, features, class_labels):
    predictions = tree.predict(features)
    
    confidence = None
    if class_labels is not None:
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == class_labels[i]:
                correct += 1
        confidence = correct / len(predictions)
    
    return predictions, confidence

