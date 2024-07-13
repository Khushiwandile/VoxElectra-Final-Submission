import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing
def load_wav_files(folder_path):
    file_paths = []
    labels = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                file_paths.append(os.path.join(root, file))
                labels.append(root.split('/')[-1])  # Assuming folder names are the labels
    return file_paths, labels

# Feature Extraction
def extract_features(file_path, max_length=128):
    y, sr = librosa.load(file_path, sr=None)
    # Extract Mel-spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or truncate the spectrogram to a fixed length
    if mel_spec_db.shape[1] < max_length:
        pad_width = max_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_length]
    
    return mel_spec_db

# Spectrogram Generation
def generate_spectrograms(file_paths, max_length=128):
    spectrograms = []
    for file_path in file_paths:
        spec = extract_features(file_path, max_length)
        spectrograms.append(spec)
    return np.array(spectrograms)

# Load Data
folder_path = 'clean_testset_wav'
file_paths, labels = load_wav_files(folder_path)
max_length = 128  # Define a fixed length for all spectrograms
spectrograms = generate_spectrograms(file_paths, max_length)

# Encode Labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = tf.keras.utils.to_categorical(encoded_labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(spectrograms, encoded_labels, test_size=0.2, random_state=42)

# Model Definition
input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Reshaping
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Model Training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Plot Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
