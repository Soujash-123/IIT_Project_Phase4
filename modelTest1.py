import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras.callbacks

# Initialize lists to hold file paths and labels
paths = []
labels = []

# Walk through the TESS directory and collect file paths and labels
for dirname, _, filenames in os.walk('./TESS'):
    for filename in filenames:
        if filename.endswith('.wav'):  # Ensure only wav files are considered
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1].split('.')[0]
            labels.append(label.lower())

# Create a DataFrame with the collected paths and labels
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels

# Plot the distribution of labels
sns.countplot(data=df, x='label')
plt.show()


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# Display waveplot and spectrogram for each emotion
emotions = df['label'].unique()
for emotion in emotions:
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectrogram(data, sampling_rate, emotion)

# Split the data into training, validation, and testing sets
X_train_paths, X_test_paths, y_train_labels, y_test_labels = train_test_split(df['speech'], df['label'], test_size=0.3, random_state=42)
X_val_paths, X_test_paths, y_val_labels, y_test_labels = train_test_split(X_test_paths, y_test_labels, test_size=1/3, random_state=42)

# Extract MFCC features for training, validation, and testing sets
X_train = np.array([extract_mfcc(path) for path in X_train_paths])
X_val = np.array([extract_mfcc(path) for path in X_val_paths])
X_test = np.array([extract_mfcc(path) for path in X_test_paths])

# Expand dimensions
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)
X_test = np.expand_dims(X_test, -1)

# One-hot encode the labels
enc = OneHotEncoder()
y_train = enc.fit_transform(np.array(y_train_labels).reshape(-1, 1)).toarray()
y_val = enc.transform(np.array(y_val_labels).reshape(-1, 1)).toarray()
y_test = enc.transform(np.array(y_test_labels).reshape(-1, 1)).toarray()

# Define the model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='auto', restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, verbose=1,
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save('speech_emotion_recognition_model.keras')

# Verify that the model is saved by listing the files in the current directory
print("Model saved as 'speech_emotion_recognition_model.keras'")
