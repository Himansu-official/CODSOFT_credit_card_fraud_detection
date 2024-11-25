import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib
import os
import logging
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./archive/training.log"),  # Save logs to a file
        logging.StreamHandler()                        # Show logs in the console
    ]
)

# Ensure models directory exists
models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

try:
    # Load preprocessed training data
    logging.info("Loading preprocessed training data...")
    train_data = pd.read_csv('./archive/preprocessed_train.csv')

    # Separate features and target
    logging.info("Separating features and target...")
    X_train = train_data.drop('is_fraud', axis=1)
    y_train = train_data['is_fraud']

    # Encode categorical columns
    logging.info("Encoding categorical features...")
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        logging.info(f"Categorical columns found: {list(categorical_cols)}")
        for col in categorical_cols:
            encoder = LabelEncoder()
            X_train[col] = encoder.fit_transform(X_train[col])

    # Handle class imbalance using SMOTE
    logging.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"Class distribution after SMOTE: {y_train_resampled.value_counts().to_dict()}")

    # Train Logistic Regression
    logging.info("Training Logistic Regression model...")
    logistic_model = LogisticRegression(random_state=42, max_iter=500)
    logistic_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(logistic_model, os.path.join(models_dir, 'logistic_model.pkl'))
    logging.info("Logistic Regression model trained and saved.")

    # Train Random Forest Classifier
    logging.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
    logging.info("Random Forest model trained and saved.")

    # Load test data for evaluation
    logging.info("Loading preprocessed test data...")
    test_data = pd.read_csv('./archive/preprocessed_test.csv')
    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    # Evaluate Logistic Regression
    logging.info("Evaluating Logistic Regression model on test data...")
    y_pred_logistic = logistic_model.predict(X_test)
    logging.info("Logistic Regression Classification Report:\n" + classification_report(y_test, y_pred_logistic))

    # Evaluate Random Forest
    logging.info("Evaluating Random Forest model on test data...")
    y_pred_rf = rf_model.predict(X_test)
    logging.info("Random Forest Classification Report:\n" + classification_report(y_test, y_pred_rf))

except Exception as e:
    logging.error(f"An error occurred: {e}")
