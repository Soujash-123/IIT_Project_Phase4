import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import keras_tuner
from tensorflow.keras import layers, models, optimizers
import joblib
from scipy.io import wavfile as wav
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
start_time = time.time()

# Load metadata and define path
metadata = pd.read_csv('urbansound8k/Book1.csv')
audio_dataset_path = 'UrbanSound8k'

# Random sample for testing
random_sample = os.path.join(audio_dataset_path, 'fold7/157940-9-0-4.wav')

# Display waveform and play audio
plt.figure(figsize=(14, 5))
data, sample_rate = librosa.load(random_sample)
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

# Feature extraction function
def feature_extraction(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # Reduced number of MFCCs
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Extract features for all audio files
extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_path = os.path.join(audio_dataset_path, 'fold' + str(row['fold']), row['slice_file_name'])
    class_label = row["class"]
    features = feature_extraction(file_path)
    extracted_features.append([features, class_label])

# Convert extracted features to DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['features', 'class'])

# Split dataset into features (X) and labels (y)
X = np.array(extracted_features_df['features'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Encode labels
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# Save the label encoder
joblib.dump(labelencoder, 'labelencoder.pkl')

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Define the model building function
def build_model(hp):
    model = tf.keras.Sequential()
    num_of_layer = hp.Int('num_of_layer', min_value=2, max_value=5, step=1)  # Increased minimum number of layers
    model.add(layers.InputLayer(input_shape=(40,)))  # Changed input shape to match the number of MFCCs
    for i in range(num_of_layer):
        model.add(layers.Dense(
            units=hp.Int(f'unit_{i}_layer', min_value=64, max_value=512, step=64),  # Increased minimum number of units
            activation='relu'
        ))
        model.add(layers.Dropout(
            rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)  # Adjusted dropout rate range
        ))
    model.add(layers.Dense(16, activation='softmax'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4])),  # Removed 1e-2 learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Hyperparameter tuning
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=50,
    max_retries_per_trial=5,
    directory='./tuner',
    project_name='Audio_classification'
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True, write_images=True),
]

tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=callbacks)

best_param = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_param)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))  # Reduced number of epochs

# Plot training history
plt.figure()
plt.plot(history.history['accuracy'], color='red', label='train_accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='val_accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(history.history['loss'], color='red', label='train_loss')
plt.plot(history.history['val_loss'], color='blue', label='val_loss')
plt.legend()
plt.grid()
plt.show()

# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
test_accuracy = test_accuracy[1] * 100
print(f"Test accuracy: {(test_accuracy)}%")

# Save the model
model.save('audio_classification_model.keras')

# Make predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=np.arange(len(labelencoder.classes_)))

# Plot confusion matrix
plt.figure(figsize=(12, 12))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelencoder.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

end_time = time.time()

# Calculate and print the total execution time
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
