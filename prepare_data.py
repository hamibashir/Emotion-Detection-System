import os
import cv2
import numpy as np
from utils import get_face_landmarks

DATA_DIR = './data'
OUTPUT_FILE = 'data.txt'
EXPECTED_FEATURE_LENGTH = 468 * 3  # 1404

def prepare_dataset():
    output = []
    classes = sorted(os.listdir(DATA_DIR))
    print(f"Detected classes: {classes}")

    for label_idx, emotion in enumerate(classes):
        emotion_dir = os.path.join(DATA_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        print(f"Processing emotion: {emotion} (label {label_idx})")

        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue

            landmarks = get_face_landmarks(image)
            if len(landmarks) != EXPECTED_FEATURE_LENGTH:
                # Skip images with no detected face or incomplete data
                continue

            landmarks.append(label_idx)
            output.append(landmarks)

    # Save all data as numpy array to text file
    if output:
        np.savetxt(OUTPUT_FILE, np.array(output))
        print(f"Data preparation completed. Saved to {OUTPUT_FILE}")
    else:
        print("No data collected. Check your images and face detection.")

if __name__ == "__main__":
    prepare_dataset()