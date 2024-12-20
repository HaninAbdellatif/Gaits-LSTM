import os
import numpy as np
import cv2
import mediapipe as mp

# Folder paths for normal and abnormal gait videos
normal_folder = r'D:\Internships\Gaits\pythonProject4\Aug\Steps AUG-20241118T050934Z-006'
abnormal_folder = r'D:\Internships\Gaits\pythonProject4\Abnormal'

# Pose estimation setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle between two vectors
def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Function to extract angles and stride features from landmarks
def get_body_angles_and_stride(landmarks):
    angles = {}
    # Right arm angles
    shoulder_r = landmarks[12]
    elbow_r = landmarks[14]
    wrist_r = landmarks[16]
    upper_arm_r = np.subtract(elbow_r, shoulder_r)
    forearm_r = np.subtract(wrist_r, elbow_r)
    angles['right_elbow'] = calculate_angle(upper_arm_r, forearm_r)

    # Right leg angles
    hip_r = landmarks[24]
    knee_r = landmarks[26]
    ankle_r = landmarks[28]
    thigh_r = np.subtract(knee_r, hip_r)
    shin_r = np.subtract(ankle_r, knee_r)
    angles['right_knee_flexion'] = calculate_angle(thigh_r, shin_r)
    angles['right_hip'] = calculate_angle(thigh_r, np.subtract(hip_r, shoulder_r))
    angles['right_ankle_dorsiflexion'] = calculate_angle(shin_r, np.subtract(ankle_r, knee_r))

    # Left arm angles
    shoulder_l = landmarks[11]
    elbow_l = landmarks[13]
    wrist_l = landmarks[15]
    upper_arm_l = np.subtract(elbow_l, shoulder_l)
    forearm_l = np.subtract(wrist_l, elbow_l)
    angles['left_elbow'] = calculate_angle(upper_arm_l, forearm_l)

    # Left leg angles
    hip_l = landmarks[23]
    knee_l = landmarks[25]
    ankle_l = landmarks[27]
    thigh_l = np.subtract(knee_l, hip_l)
    shin_l = np.subtract(ankle_l, knee_l)
    angles['left_knee_flexion'] = calculate_angle(thigh_l, shin_l)
    angles['left_hip'] = calculate_angle(thigh_l, np.subtract(hip_l, shoulder_l))
    angles['left_ankle_dorsiflexion'] = calculate_angle(shin_l, np.subtract(ankle_l, knee_l))

    # Trunk inclination angle
    trunk_vector_l = np.subtract(hip_l, shoulder_l)
    trunk_vector_r = np.subtract(hip_r, shoulder_r)
    angles['trunk_inclination'] = calculate_angle(trunk_vector_l, trunk_vector_r)

    # Stride Length and Width
    stride_length = calculate_distance(ankle_r, ankle_l)
    stride_width = calculate_distance(hip_r, hip_l)
    angles['stride_length'] = stride_length
    angles['stride_width'] = stride_width

    return angles


# Function to extract landmarks and angles from video and save them
def process_video_and_save_landmarks_and_angles(video_path, save_landmarks_path, save_angles_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarks_list = []
    angles_list = []

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with mediapipe pose model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            landmarks_list.append(landmarks)
            angles = get_body_angles_and_stride(landmarks)
            angles_list.append(angles)

        # Show progress every 10 frames
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num} / {frame_count} in {video_path}")

    cap.release()

    # Save landmarks and angles to .npy files
    np.save(save_landmarks_path, landmarks_list)
    np.save(save_angles_path, angles_list)
    print(f"Landmarks saved to {save_landmarks_path}")
    print(f"Angles saved to {save_angles_path}")

    return landmarks_list, angles_list


# Function to extract features from each video
def extract_features_and_save(folder, label):
    features = []
    labels = []
    print(f"Extracting features from {folder}...")
    for file in os.listdir(folder):
        if file.endswith('.mp4'):  # Only process .mp4 files
            video_path = os.path.join(folder, file)
            video_name = os.path.splitext(file)[0]  # Get the video name without extension
            save_landmarks_path = os.path.join(folder, f'{video_name}_landmarks.npy')
            save_angles_path = os.path.join(folder, f'{video_name}_angles.npy')

            landmarks, angles = process_video_and_save_landmarks_and_angles(video_path, save_landmarks_path, save_angles_path)

            # For each step in the video, calculate features
            for step_landmarks, step_angles in zip(landmarks, angles):
                features.append(list(step_angles.values()))
                labels.append(label)

            print(f"Finished extracting features from {video_path}")

    return np.array(features), np.array(labels)


# Load data from both normal and abnormal folders
normal_features, normal_labels = extract_features_and_save(normal_folder, label=0)  # 0 for normal
#abnormal_features, abnormal_labels = extract_features_and_save(abnormal_folder, label=1)  # 1 for abnormal

# Combine normal and abnormal data
X = np.vstack((normal_features, abnormal_features))
y = np.concatenate((normal_labels, abnormal_labels))

# Now X and y are ready for training with an LSTM or other model
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
