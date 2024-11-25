import pandas as pd

# Correct file paths (remove 'archive/')
train_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')

# Display first few rows
print("Train Data:")
print(train_data.head())

print("\nTest Data:")
print(test_data.head())

# Data structure and missing values
print("\nTrain Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())

print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())

# Analyze target variable
print("\nTarget Variable Distribution in Train Data:")
print(train_data['is_fraud'].value_counts())
