import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory: location of the preprocess.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
TRAIN_PATH = os.path.join(BASE_DIR, 'fraudTrain.csv')
TEST_PATH = os.path.join(BASE_DIR, 'fraudTest.csv')
OUTPUT_TRAIN = os.path.join(BASE_DIR, 'preprocessed_train.csv')
OUTPUT_TEST = os.path.join(BASE_DIR, 'preprocessed_test.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')  # Ensure results folder exists

def preprocess_data():
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Load datasets
    print("Loading datasets...")
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    # Drop irrelevant columns
    drop_columns = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num']
    train_data.drop(columns=drop_columns, inplace=True)
    test_data.drop(columns=drop_columns, inplace=True)

    # Combine train and test for consistent encoding
    combined_data = pd.concat([train_data, test_data], axis=0)

    # Encode categorical columns
    print("Encoding categorical features...")
    categorical_columns = ['merchant', 'category', 'gender', 'job']
    label_encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        combined_data[col] = encoder.fit_transform(combined_data[col])
        label_encoders[col] = encoder

    # Separate train and test datasets
    train_data = combined_data.iloc[:len(train_data)]
    test_data = combined_data.iloc[len(train_data):]

    # Scale numerical features
    print("Scaling numerical features...")
    numeric_columns = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    scaler = StandardScaler()
    train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])
    test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])

    # Handle class imbalance using SMOTE
    print("Handling class imbalance using SMOTE...")
    X_train = train_data.drop(columns=['is_fraud'])
    y_train = train_data['is_fraud']

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Visualize class distribution before and after SMOTE
    print("Saving class distribution plots...")
    plt.figure(figsize=(10, 5))
    sns.countplot(x='is_fraud', data=train_data)
    plt.title("Class Distribution Before SMOTE")
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution_before_smote.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.countplot(x=y_train_resampled)
    plt.title("Class Distribution After SMOTE")
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution_after_smote.png'))
    plt.close()

    # Save preprocessed data
    print("Saving preprocessed data...")
    train_preprocessed = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    train_preprocessed['is_fraud'] = y_train_resampled
    train_preprocessed.to_csv(OUTPUT_TRAIN, index=False)
    test_data.to_csv(OUTPUT_TEST, index=False)

    print("Preprocessing completed. Files saved:")
    print(f"- {OUTPUT_TRAIN}")
    print(f"- {OUTPUT_TEST}")
    print("\nPlots saved to the 'results' directory.")

# Execute preprocessing
if __name__ == '__main__':
    preprocess_data()
