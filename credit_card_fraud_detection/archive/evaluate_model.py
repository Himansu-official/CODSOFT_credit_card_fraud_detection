import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Define file paths
test_data_path = './archive/preprocessed_test.csv'
logistic_model_path = './models/logistic_model.pkl'
rf_model_path = './models/random_forest_model.pkl'
roc_curve_path = './archive/results/roc_curve.png'
conf_matrix_path = './archive/results/confusion_matrix.png'

# Check if files exist
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test dataset not found at {test_data_path}")
if not os.path.exists(logistic_model_path):
    raise FileNotFoundError(f"Logistic Regression model not found at {logistic_model_path}")
if not os.path.exists(rf_model_path):
    raise FileNotFoundError(f"Random Forest model not found at {rf_model_path}")

# Load preprocessed test dataset
print("Loading test dataset...")
test_data = pd.read_csv(test_data_path)

# Separate features and target
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# Load models
print("Loading models...")
logistic_model = joblib.load(logistic_model_path)
rf_model = joblib.load(rf_model_path)

# Evaluate Logistic Regression
print("Evaluating Logistic Regression model...")
y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_logistic))

# Evaluate Random Forest
print("Evaluating Random Forest model...")
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Plot ROC Curve for Random Forest
print("Plotting ROC Curve...")
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(roc_curve_path)
plt.show()

# Save confusion matrix
print("Saving Confusion Matrix...")
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure()
plt.matshow(conf_matrix, cmap='Pastel1')
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(conf_matrix_path)
plt.show()

print("Evaluation completed. Results saved to 'results' directory.")
