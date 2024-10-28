import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv("processed_cryptography_dataset.csv")  # Adjust the path to your dataset

# Display initial DataFrame info
print("Initial DataFrame:")
print(df.info())

# Drop 'private_key' column if it's entirely null
if df['private_key'].isnull().all():
    print("Dropping 'private_key' column as it contains all NaN values.")
    df = df.drop(columns=['private_key'])

# Check if any rows remain after dropping
print(f"Rows after dropping 'private_key': {df.shape[0]}")

# Convert 'ciphertext' to numeric, forcing errors to NaN
df['ciphertext'] = pd.to_numeric(df['ciphertext'], errors='coerce')
print("After converting 'ciphertext' to numeric:")
print(df['ciphertext'].unique())  # Display unique values to understand the data

# Check for NaN values in features
print("NaN values in features:")
print(df[['plaintext', 'ciphertext']].isnull().sum())

# Drop rows with NaN values in 'ciphertext' after conversion
df = df.dropna(subset=['ciphertext'])
print(f"After dropping NaN in 'ciphertext': {df.shape[0]} rows remaining.")

# Final check for NaN values
print("Final NaN values before split:")
print(df[['plaintext', 'ciphertext']].isnull().sum())

# Prepare features and target variable
X = df[['plaintext', 'ciphertext']]
y = df['algorithm']

# Check shapes before splitting
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Convert categorical text data to numerical data
X = pd.get_dummies(X, columns=['plaintext'], drop_first=True)

# Check shape after get_dummies
print(f"X shape after get_dummies: {X.shape}")

# Scale the 'ciphertext' column if there are any rows left
if not X.empty:
    scaler = MinMaxScaler()
    X[['ciphertext']] = scaler.fit_transform(X[['ciphertext']])
else:
    print("X is empty after processing, cannot scale.")

# Split the dataset into training and testing sets only if there are rows
if not X.empty and not y.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
else:
    print("Not enough data to train the model.")
