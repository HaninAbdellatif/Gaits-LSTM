# Gait Analysis for Normal vs. Abnormal Classification Using LSTM

## Overview

This project implements a deep learning-based approach for classifying gait sequences as either "Normal" or "Abnormal" using a Long Short-Term Memory (LSTM) network. The model analyzes body joint angles and stride parameters extracted from video sequences of human walking. By learning from these gait patterns, the model can identify whether the gait is normal or indicative of an abnormality, which can be useful in clinical diagnostics or assistive technology applications for people with movement disorders.

## Goal

The goal of this project is to:
1. Process raw gait data (extracted joint angles and landmarks from gait videos).
2. Train a machine learning model (LSTM) to classify sequences of body movements into normal and abnormal gait patterns.
3. Provide a mechanism to predict gait abnormalities from new sequences without retraining the model.

## Approach

The approach is divided into two main parts:
1.**Data Splitting**: Every video was split into individual steps and organise them in a sequence of step to process steps instead of a whole gait

2. **Data Processing**: Gait sequences are processed by extracting body joint angles (like knee flexion, ankle dorsiflexion) and stride parameters (stride length, width). These features were extracted from each step in a gait sequence. These sequences are padded to a fixed length (110 steps) for consistent input into the model.
   
3. **Modeling**: We use an LSTM-based deep learning model to classify the gait sequences. The LSTM network is well-suited for sequence prediction tasks, as it can learn from the temporal dependencies in the gait patterns.

The model is trained on labeled gait sequences (Normal vs. Abnormal), and it predicts the class for any given sequence of steps.

## Data

The dataset consists of video sequences captured during gait analysis. Each video sequence corresponds to a gait, and for each step in the gait sequence, body joint angles and landmark coordinates are extracted. The data is structured as follows:

- **Angle Data**: Includes joint angles (e.g., knee, ankle, hip flexion) and stride parameters (stride length, width) for each step.
- **Landmark Data**: Contains the body landmark coordinates (e.g., the positions of body joints in 3D space) for each step.
- **Folders**: The data is organized into `Training` and `Testing` which both includes `Normal` and `Abnormal` folders, with each folder containing gait sequences stored in subfolders. Each subfolder corresponds to a unique gait sequence, and within each subfolder, files are named in the format `videonumber_angles.npy` for angle data and `videonumber_landmarks.npy` for landmark data.

## Key Features

1. **Data Preprocessing**:
   - Sequences are padded to a maximum length (default: 110 steps) to ensure uniform input size.
   - Angle data is extracted from JSON-like dictionaries into numerical arrays that represent joint angles and stride parameters for each step.
   - Landmark data is loaded and used for processing (although the model currently focuses on angle features for classification).

2. **Model Architecture**:
   - **LSTM Network**: The core of the model is a two-layer LSTM network designed to process sequential data. The LSTM layers capture the temporal dependencies between steps in the gait sequence.
   - **Dense Layer**: After the LSTM layers, a dense layer with ReLU activation is used to process the features before the final classification layer.
   - **Output Layer**: The final output layer uses a softmax activation to classify the gait sequence into one of two categories: Normal or Abnormal.

3. **Prediction**:
   - Once the model is trained, it can predict the label (Normal/Abnormal) for new gait sequences by analyzing the temporal patterns in the joint angles and stride data.

## How to Use
### 2. Extract data
From every sequence of data exists in a folder extract landmarks of the 3D pose estimation and then the joints angels and store them in two files to use the data extracted in them in the model
### 1. Training the Model

To train the model, ensure that you have the gait sequence data in the correct format and placed in the appropriate folders.

- Place the gait sequence data in the `Training` folder, divided into `Normal` and `Abnormal` subfolders, and for testing do the same with `Testing` folder.
- Run the `train_model.py` script to train the model. The script will load the data, preprocess it, and train an LSTM model. After training, the model will be saved as `gait_model.h5`.


