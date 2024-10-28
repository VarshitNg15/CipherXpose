import pandas as pd

# Load the dataset
csv_file_name = "cryptography_dataset.csv"
df = pd.read_csv(csv_file_name)

# Display the initial DataFrame
print("Initial DataFrame:")
print(df)

# Check for missing values
print("\nMissing values in the DataFrame:")
print(df.isnull().sum())

# Drop rows with missing algorithm values (since these are not useful for training)
df.dropna(subset=['algorithm'], inplace=True)

# Display DataFrame after dropping missing values
print("\nDataFrame after dropping missing values:")
print(df)

# Convert plaintext from hex string to integer
df['plaintext'] = df['plaintext'].apply(lambda x: int(x, 16))  # Convert hex string to int

# Function to safely extract the ciphertext
def safe_extract_ciphertext(ciphertext):
    parts = ciphertext.split(':')
    if len(parts) == 2:
        return int(parts[1], 16)  # Return the integer value of the ciphertext part
    else:
        return None  # Return None for entries with unexpected format

# Apply the function to the ciphertext column
df['ciphertext'] = df['ciphertext'].apply(safe_extract_ciphertext)

# Drop rows where the ciphertext extraction failed
df.dropna(subset=['ciphertext'], inplace=True)

# Display the processed DataFrame
print("\nProcessed DataFrame:")
print(df)

# Save the processed DataFrame to a new CSV file
processed_csv_file_name = "processed_cryptography_dataset.csv"
df.to_csv(processed_csv_file_name, index=False)

print(f"\nProcessed dataset saved to '{processed_csv_file_name}'.")