import cv2
import numpy as np
import os

dataset_path = r"D:\SignatureVerification\dataset"

def load_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {filename}")
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load {img_path}. Skipping this file.")
            continue

        try:
            img = cv2.resize(img, (50, 50))
        except Exception as e:
            print(f"Error resizing {img_path}: {e}")
            continue

        img = img.flatten() / 255.0  # Normalize
        images.append(img)
        labels.append(label)
    
    print(f"Loaded {len(images)} images from {folder_path}")
    return images, labels

def prepare_dataset(dataset_path):
    real_images, real_labels = load_images(os.path.join(dataset_path, 'real'), 1)
    fake_images, fake_labels = load_images(os.path.join(dataset_path, 'fake'), 0)

    X = np.array(real_images + fake_images)
    y = np.array(real_labels + fake_labels).reshape(-1, 1)

    print(f"Dataset Loaded: {X.shape[0]} samples")  # ✅ Correct f-string
    return X, y
