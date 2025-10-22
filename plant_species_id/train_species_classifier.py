#!/usr/bin/env python3
"""
Train Species Classifier for Plant Identification
Trains a machine learning model to classify plant species from leaf images
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import logging

# Import our leaf detection functions from algo1.ipynb
# We'll extract the key functions and put them here for modularity

def preprocess_image(image):
    """Preprocess image for feature extraction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = cv2.equalizeHist(norm)
    return enhanced

def custom_edge_map(gray):
    """Extract edge map using custom gradient detection"""
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    gx = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kx)
    gy = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, ky)
    mag = np.sqrt(gx**2 + gy**2)
    mag_u8 = ((mag / (mag.max() + 1e-12)) * 255).astype(np.uint8)
    _, edge_binary = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edge_binary, mag_u8

def segment_largest_object(image, gray):
    """Segment largest object (assumed to be leaf)"""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    if np.count_nonzero(cleaned) < 200:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 30, 20])
        upper_green = np.array([100, 255, 255])
        color_mask = cv2.inRange(hsv, lower_green, upper_green)
        cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, cleaned
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask, largest, cleaned

def extract_features(image, mask, contour):
    """Extract comprehensive features from leaf image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_area = image.shape[0] * image.shape[1]

    mask_pixels = np.count_nonzero(mask) if mask is not None else 0
    contour_area = cv2.contourArea(contour) if contour is not None else 0
    area_ratio = contour_area / image_area if image_area > 0 else 0
    mask_coverage = mask_pixels / image_area if image_area > 0 else 0

    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    edge_density = edge_count / image_area if image_area > 0 else 0

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    _, bin_mask = cv2.threshold(masked_gray, 1, 255, cv2.THRESH_BINARY)
    from skimage.morphology import skeletonize
    skel = skeletonize(bin_mask // 255)
    skeleton_pixels = np.count_nonzero(skel)
    vein_density = skeleton_pixels / mask_pixels if mask_pixels > 0 else 0

    masked_gray_vals = gray[mask > 0]
    gray_variance = float(np.var(masked_gray_vals)) if masked_gray_vals.size > 0 else 0.0
    from skimage.measure import shannon_entropy
    gray_entropy = float(shannon_entropy(masked_gray_vals)) if masked_gray_vals.size > 0 else 0.0

    if mask_pixels > 0:
        masked_pixels = image[mask > 0].reshape(-1, 3)
        color_std = float(np.mean(np.std(masked_pixels, axis=0)))
    else:
        color_std = 0.0

    solidity = 0.0
    if contour is not None:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = contour_area / hull_area

    # Geometric features
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        rect_area = w * h
        extent = float(contour_area) / rect_area if rect_area > 0 else 0
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
    else:
        aspect_ratio = extent = 0
        hu_moments = np.zeros(7)

    feats = {
        'image_area': image_area,
        'contour_area': contour_area,
        'mask_pixels': mask_pixels,
        'area_ratio': area_ratio,
        'mask_coverage': mask_coverage,
        'edge_count': edge_count,
        'edge_density': edge_density,
        'vein_density': vein_density,
        'skeleton_pixels': skeleton_pixels,
        'gray_variance': gray_variance,
        'gray_entropy': gray_entropy,
        'color_std': color_std,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'hu_moment_1': hu_moments[0],
        'hu_moment_2': hu_moments[1],
        'hu_moment_3': hu_moments[2],
        'hu_moment_4': hu_moments[3],
        'hu_moment_5': hu_moments[4],
        'hu_moment_6': hu_moments[5],
        'hu_moment_7': hu_moments[6]
    }
    return feats

def load_dataset(data_dir):
    """Load dataset from directory structure"""
    data_path = Path(data_dir)
    images = []
    labels = []
    file_paths = []

    species_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    for species_dir in tqdm(species_dirs, desc="Loading dataset"):
        species_name = species_dir.name

        image_files = list(species_dir.glob('*.jpg')) + \
                     list(species_dir.glob('*.png')) + \
                     list(species_dir.glob('*.jpeg'))

        print(f"Loading {len(image_files)} images for {species_name}")

        for img_file in image_files:
            try:
                image = cv2.imread(str(img_file))
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                labels.append(species_name)
                file_paths.append(str(img_file))

            except Exception as e:
                print(f"Failed to load {img_file}: {e}")

    return images, labels, file_paths

def extract_features_from_dataset(images):
    """Extract features from all images in dataset"""
    features_list = []

    for img in tqdm(images, desc="Extracting features"):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask, contour, _ = segment_largest_object(img, gray)

            if mask is None or contour is None:
                # Use zero features for failed segmentation
                features = {k: 0.0 for k in ['image_area', 'contour_area', 'mask_pixels', 'area_ratio',
                                           'mask_coverage', 'edge_count', 'edge_density', 'vein_density',
                                           'skeleton_pixels', 'gray_variance', 'gray_entropy', 'color_std',
                                           'solidity', 'aspect_ratio', 'extent', 'hu_moment_1', 'hu_moment_2',
                                           'hu_moment_3', 'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7']}
            else:
                features = extract_features(img, mask, contour)

            # Convert to feature vector
            feature_vector = np.array(list(features.values()))
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            features_list.append(feature_vector)

        except Exception as e:
            print(f"Feature extraction failed: {e}")
            features_list.append(np.zeros(20))  # Placeholder

    return np.array(features_list)

def train_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest classifier"""
    print("Training Random Forest classifier...")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(".3f")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, accuracy

def main():
    """Main training function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load dataset
    print("Loading dataset...")
    images, labels, file_paths = load_dataset("data")

    if len(images) == 0:
        print("No images found in dataset!")
        return

    print(f"Loaded {len(images)} images from {len(set(labels))} species")

    # Extract features
    print("Extracting features...")
    features = extract_features_from_dataset(images)

    print(f"Feature matrix shape: {features.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train classifier
    classifier, accuracy = train_classifier(X_train, y_train, X_test, y_test)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/species_classifier.joblib"
    joblib.dump(classifier, model_path)
    print(f"Model saved to {model_path}")

    # Save label classes
    label_classes = list(set(labels))
    np.save("models/label_classes.npy", label_classes)
    print(f"Label classes saved: {label_classes}")

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
