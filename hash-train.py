
# Install required libraries (if needed, but TensorFlow and Scikit-learn are pre-installed in Colab)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import joblib
from google.colab import drive  # For Google Drive access

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Define the path in Google Drive where you want to save the models
drive_path = '/content/drive/My Drive/hash_model/'

# Load and prepare the dataset
# Check if the file exists in the specified location and verify its name for typos
df = pd.read_csv('/content/drive/My Drive/hashing_algorithms_dataset_colab.csv')  # Make sure the file is available and the name is correct

# Extract features and labels
X = df[['Mean', 'StdDev', 'Entropy', 'Kurtosis', 'Skewness', 'MinValue', 'MaxValue', 'Range', 'UniqueByteCount']].values
y = df['Label'].values

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for CNN (assume single-channel images)
X_train = X_train.reshape(-1, 9, 1, 1)  # Reshape to (samples, height, width, channels)
X_test = X_test.reshape(-1, 9, 1, 1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(9, 1, 1)))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 2: Save the model and preprocessing tools in Google Drive
model.save(drive_path + 'hashing_algorithm_cnn_model.h5')
joblib.dump(label_encoder, drive_path + 'label_encoder.pkl')
joblib.dump(scaler, drive_path + 'scaler.pkl')

print("Model and preprocessing tools saved in Google Drive at: ", drive_path)
