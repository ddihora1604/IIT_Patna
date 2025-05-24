import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, cohen_kappa_score, log_loss,
                             mean_squared_error, mean_absolute_error, r2_score)
import time
import os

# Set the number of K folds as a global variable
K_FOLDS = 2

# Read the dataset from CSV file
df = pd.read_csv('data.csv')

# Take 20% of the data
df = df.sample(frac=0.2, random_state=42)

# Rename the last column as 'label'
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

X = df.drop(columns=['label']).values
y = df['label'].values

timing_results = []


# Define classifiers and metrics
classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
}
# Store results
results = []

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Create a directory to save confusion matrices
os.makedirs("confusion_matrices", exist_ok=True)

# Helper function for confusion matrix metrics
def confusion_matrix_metrics(cm, classes):
    metrics = {}
    for idx, class_label in enumerate(classes):
        TP = cm[idx, idx]  # True Positives for this class
        FP = cm[:, idx].sum() - TP  # False Positives for this class
        FN = cm[idx, :].sum() - TP  # False Negatives for this class
        TN = cm.sum() - (TP + FP + FN)  # True Negatives for this class

        metrics[class_label] = {
            'TPR': TP / (TP + FN + 1e-10) if (TP + FN) > 0 else 0,
            'TNR': TN / (TN + FP + 1e-10) if (TN + FP) > 0 else 0,
            'FPR': FP / (FP + TN + 1e-10) if (FP + TN) > 0 else 0,
            'FNR': FN / (FN + TP + 1e-10) if (FN + TP) > 0 else 0
        }
    return metrics

# Iterate over classifiers
for clf_name, clf in classifiers.items():
    fold_idx = 1
    for train_index, test_index in kf.split(X):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Record start time
        start_train_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_train_time

        start_test_time = time.time()
        y_pred = clf.predict(X_test)
        test_time = time.time() - start_test_time

        timing_results.append({
            'Classifier': clf_name,
            'Fold': fold_idx,
            'Training Time (s)': train_time,
            'Testing Time (s)': test_time,
            'Total Time (s)': train_time + test_time
        })

        # Compute metrics
        unique_classes = np.unique(y)
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        cm_metrics = confusion_matrix_metrics(cm, unique_classes)

        class_metrics_list = []

        for class_label in unique_classes:
            class_mask = (y_test == class_label)
            if class_mask.sum() == 0:
                # Skip classes with no instances in the test set for this fold
                class_specific_metrics = {
                    'Classifier': clf_name,
                    'Fold': fold_idx,
                    'Class': class_label,
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1 Score': np.nan,
                    'Balanced Accuracy': np.nan,
                    'True Positive Rate (TPR)': np.nan,
                    'True Negative Rate (TNR)': np.nan,
                    'False Positive Rate (FPR)': np.nan,
                    'False Negative Rate (FNR)': np.nan,
                    'Training Time (s)': train_time,
                    'Testing Time (s)': test_time
                }
            else:
                class_specific_metrics = {
                    'Classifier': clf_name,
                    'Fold': fold_idx,
                    'Class': class_label,
                    'Accuracy': accuracy_score(y_test[class_mask], y_pred[class_mask]) if np.any(class_mask) else np.nan,
                    'Precision': precision_score(y_test[class_mask], y_pred[class_mask], average='weighted', zero_division=0) if np.any(class_mask) else np.nan,
                    'Recall': recall_score(y_test[class_mask], y_pred[class_mask], average='weighted') if np.any(class_mask) else np.nan,
                    'F1 Score': f1_score(y_test[class_mask], y_pred[class_mask], average='weighted') if np.any(class_mask) else np.nan,
                    'Balanced Accuracy': balanced_accuracy_score(y_test[class_mask], y_pred[class_mask]) if np.any(class_mask) else np.nan,
                    'True Positive Rate (TPR)': cm_metrics[class_label]['TPR'],
                    'True Negative Rate (TNR)': cm_metrics[class_label]['TNR'],
                    'False Positive Rate (FPR)': cm_metrics[class_label]['FPR'],
                    'False Negative Rate (FNR)': cm_metrics[class_label]['FNR'],
                    'Training Time (s)': train_time,
                    'Testing Time (s)': test_time
                }

            class_metrics_list.append(class_specific_metrics)

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
        plt.title(f"{clf_name} - Fold {fold_idx} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrices/{clf_name}_fold_{fold_idx}.png")
        plt.close()

        # Append results for this fold
        results.extend(class_metrics_list)
        fold_idx += 1

timing_df = pd.DataFrame(timing_results)
timing_df.to_csv("time.csv", index=False)

# Create a DataFrame for results
results_df = pd.DataFrame(results)
print("Classification Metrics Across Folds:")
print(results_df)

# Save results to CSV
results_df.to_csv("metrics.csv", index=False)