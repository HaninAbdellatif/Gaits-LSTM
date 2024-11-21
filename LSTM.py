import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Define the function to load and process the gait sequences
def load_gait_sequences_from_folder(folder_path, label=0, max_steps=110):
    all_sequences = []
    all_labels = []

    # Define the expected order of features
    angle_keys = [
        'right_elbow', 'right_knee_flexion', 'right_hip', 'right_ankle_dorsiflexion',
        'left_elbow', 'left_knee_flexion', 'left_hip', 'left_ankle_dorsiflexion',
        'trunk_inclination', 'stride_length', 'stride_width'
    ]

    # Iterate through the subfolders in the folder (these represent different gait sequences)
    for sequence_folder in os.listdir(folder_path):
        sequence_path = os.path.join(folder_path, sequence_folder)

        # Only process directories (which represent each sequence)
        if os.path.isdir(sequence_path):
            all_sequence_steps = []

            # Iterate through the files in the sequence folder
            for file_name in os.listdir(sequence_path):
                if file_name.endswith('_angles.npy'):  # This identifies the angle files
                    print(f"Found file: {file_name} in {sequence_folder}")

                    # Extract the corresponding landmarks file
                    base_name = file_name.replace('_angles.npy', '')
                    landmarks_file = base_name + '_landmarks.npy'

                    # Check if both angle and landmark files exist
                    if os.path.exists(os.path.join(sequence_path, landmarks_file)):
                        angles_data = np.load(os.path.join(sequence_path, file_name), allow_pickle=True)
                        landmarks_data = np.load(os.path.join(sequence_path, landmarks_file), allow_pickle=True)

                        # Check if angles_data is empty
                        if angles_data.size == 0:
                            print(f"File {file_name} is empty. Skipping...")
                            continue

                        # Convert angles_data to numeric array if it contains dictionaries
                        if isinstance(angles_data[0], dict):
                            angles_data = np.array([  # Flatten dictionary to array
                                [step_data[key] for key in angle_keys] for step_data in angles_data
                            ])

                        print(f"Initial angles data for file {file_name}: {angles_data[:3]}")  # Display the first 3 steps

                        # Store the sequence data for the current sequence
                        all_sequence_steps.append(angles_data)
                    else:
                        print(f"Landmark file not found for {file_name}. Skipping...")

            # Pad the sequences for the current folder
            for angles_data in all_sequence_steps:
                if angles_data.shape[0] < max_steps:
                    padding = np.zeros((max_steps - angles_data.shape[0], angles_data.shape[1]))
                    padded_angles = np.concatenate((angles_data, padding), axis=0)
                else:
                    padded_angles = angles_data[:max_steps, :]

                all_sequences.append(padded_angles)
                all_labels.append(label)

    # Convert lists to numpy arrays
    return np.array(all_sequences), np.array(all_labels)

# Paths to the gait folders
training_folder = r'D:\Internships\Gaits\pythonProject4\Training'
testing_folder = r'D:\Internships\Gaits\pythonProject4\Testing'

# Load data from the Training set
X_train_normal, y_train_normal = load_gait_sequences_from_folder(os.path.join(training_folder, 'Normal'), label=0)
X_train_abnormal, y_train_abnormal = load_gait_sequences_from_folder(os.path.join(training_folder, 'Abnormal'), label=1)

# Load data from the Testing set
X_test_normal, y_test_normal = load_gait_sequences_from_folder(os.path.join(testing_folder, 'Normal'), label=0)
X_test_abnormal, y_test_abnormal = load_gait_sequences_from_folder(os.path.join(testing_folder, 'Abnormal'), label=1)

# Combine and split into train/test sets
X_train = np.concatenate((X_train_normal, X_train_abnormal), axis=0)
y_train = np.concatenate((y_train_normal, y_train_abnormal), axis=0)
X_test = np.concatenate((X_test_normal, X_test_abnormal), axis=0)
y_test = np.concatenate((y_test_normal, y_test_abnormal), axis=0)

# Check shapes before reshaping
print(f"X_train shape before reshaping: {X_train.shape}")
print(f"X_test shape before reshaping: {X_test.shape}")

# Reshape data to fit the LSTM input shape (samples, time steps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 11))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 11))

print(f"First two entries in X_train_lstm: {X_train_lstm[:2]}")

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_lstm, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model to a file
model.save('gait_model.h5')  # The '.h5' extension is commonly used for Keras models




