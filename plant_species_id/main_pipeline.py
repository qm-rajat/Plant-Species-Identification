#!/usr/bin/env python3
"""
Unified Plant Species Identification Pipeline
Combines leaf detection with species classification
"""

import os
import cv2
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional

# Import leaf detection functions from algo1.ipynb
from skimage.morphology import skeletonize
from skimage.measure import shannon_entropy

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
    skel = skeletonize(bin_mask // 255)
    skeleton_pixels = np.count_nonzero(skel)
    vein_density = skeleton_pixels / mask_pixels if mask_pixels > 0 else 0

    masked_gray_vals = gray[mask > 0]
    gray_variance = float(np.var(masked_gray_vals)) if masked_gray_vals.size > 0 else 0.0
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

def is_leaf_detector(image, debug=False) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """Detect if image contains a leaf"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask, contour, _ = segment_largest_object(image, gray)

    if mask is None or contour is None:
        return False, {'reason': 'No object detected'}, {}

    feats = extract_features(image, mask, contour)

    # heuristics
    votes = {
        'enough_size': feats['contour_area'] > 1000,
        'not_full_frame': feats['area_ratio'] < 0.95,
        'has_edges': feats['edge_count'] > 500,
        'has_veins': feats['vein_density'] > 0.005,
        'has_variation': feats['gray_variance'] > 50,
        'has_entropy': feats['gray_entropy'] > 2.5,
        'color_varied': feats['color_std'] > 5
    }
    score = sum(votes.values())

    # base decision
    is_leaf = votes['has_edges'] and votes['has_entropy'] and votes['enough_size']

    if score < 4:
        is_leaf = False

    # safeguard: flat green wall rejection
    if feats['vein_density'] < 0.001 and feats['solidity'] > 0.9 and feats['gray_entropy'] < 3.0:
        is_leaf = False

    if feats['gray_entropy'] < 2.8 and feats['edge_count'] < 300:
        is_leaf = False

    confidence = float(score) / len(votes)
    if not is_leaf:
        confidence *= 0.5

    failed_reasons = [k for k, v in votes.items() if not v]
    diagnostics = {**feats,
                'votes': votes,
                'score': score,
                'is_leaf': is_leaf,
                'failed_reasons': failed_reasons,
                'confidence': confidence}

    if debug:
        print("Diagnostics: ", diagnostics )
        for key,value in diagnostics.items():
            print(f"{key} :{value}")
        print("Leaf?", is_leaf)
        print(" ----------------------------- ")
    return is_leaf, diagnostics, feats

class PlantSpeciesClassifier:
    """Wrapper for the trained species classification model"""

    def __init__(self, model_path: str = "models/species_classifier.joblib",
                 labels_path: str = "models/label_classes.npy"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.label_classes = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and label classes"""
        try:
            self.model = joblib.load(self.model_path)
            self.label_classes = np.load(self.labels_path, allow_pickle=True)
            print(f"Model loaded successfully. Classes: {self.label_classes}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {e}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """Predict species from features"""
        if self.model is None:
            raise Exception("Model not loaded")

        # Convert features dict to feature vector (same order as training)
        feature_keys = ['image_area', 'contour_area', 'mask_pixels', 'area_ratio',
                       'mask_coverage', 'edge_count', 'edge_density', 'vein_density',
                       'skeleton_pixels', 'gray_variance', 'gray_entropy', 'color_std',
                       'solidity', 'aspect_ratio', 'extent', 'hu_moment_1', 'hu_moment_2',
                       'hu_moment_3', 'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7']

        feature_vector = np.array([features.get(key, 0.0) for key in feature_keys])
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Get prediction probabilities
        probabilities = self.model.predict_proba([feature_vector])[0]

        # Get the predicted class and confidence
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.label_classes[predicted_idx]
        confidence = probabilities[predicted_idx]

        return predicted_class, float(confidence)

def identify_plant_species(image_path: str, debug: bool = False) -> Dict[str, Any]:
    """
    Complete pipeline: detect leaf and identify species

    Args:
        image_path: Path to the image file
        debug: Enable debug output

    Returns:
        Dictionary with identification results
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load image
    if not os.path.exists(image_path):
        return {
            'success': False,
            'error': f'Image file not found: {image_path}',
            'is_leaf': False,
            'species': None,
            'confidence': 0.0
        }

    try:
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Could not load image: {image_path}',
                'is_leaf': False,
                'species': None,
                'confidence': 0.0
            }

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

    except Exception as e:
        return {
            'success': False,
            'error': f'Error loading image: {str(e)}',
            'is_leaf': False,
            'species': None,
            'confidence': 0.0
        }

    # Step 1: Leaf detection
    try:
        is_leaf, leaf_diagnostics, features = is_leaf_detector(image, debug=debug)
    except Exception as e:
        logger.error(f"Leaf detection failed: {e}")
        return {
            'success': False,
            'error': f'Leaf detection failed: {str(e)}',
            'is_leaf': False,
            'species': None,
            'confidence': 0.0
        }

    if not is_leaf:
        return {
            'success': True,
            'is_leaf': False,
            'species': None,
            'confidence': 0.0,
            'leaf_diagnostics': leaf_diagnostics,
            'message': 'Image does not contain a recognizable leaf'
        }

    # Step 2: Species classification
    try:
        classifier = PlantSpeciesClassifier()
        species, species_confidence = classifier.predict(features)
    except Exception as e:
        logger.error(f"Species classification failed: {e}")
        return {
            'success': False,
            'error': f'Species classification failed: {str(e)}',
            'is_leaf': True,
            'species': None,
            'confidence': 0.0,
            'leaf_diagnostics': leaf_diagnostics
        }

    # Return successful result
    return {
        'success': True,
        'is_leaf': True,
        'species': species,
        'confidence': species_confidence,
        'leaf_diagnostics': leaf_diagnostics,
        'features': features,
        'message': f'Identified as {species} with {species_confidence:.2%} confidence'
    }

def main():
    """Command-line interface for plant species identification"""
    import argparse

    parser = argparse.ArgumentParser(description='Identify plant species from leaf images')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    result = identify_plant_species(args.image_path, debug=args.debug)

    print("\n=== Plant Species Identification Result ===")
    print(f"Image: {args.image_path}")
    print(f"Success: {result['success']}")

    if not result['success']:
        print(f"Error: {result['error']}")
        return

    print(f"Is Leaf: {result['is_leaf']}")

    if result['is_leaf']:
        print(f"Species: {result['species']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Message: {result['message']}")
    else:
        print(f"Message: {result['message']}")

    if args.debug and 'leaf_diagnostics' in result:
        print("\n--- Leaf Detection Diagnostics ---")
        diag = result['leaf_diagnostics']
        print(f"Confidence: {diag.get('confidence', 0):.2f}")
        print(f"Score: {diag.get('score', 0)}/7")
        if 'failed_reasons' in diag and diag['failed_reasons']:
            print(f"Failed criteria: {', '.join(diag['failed_reasons'])}")

if __name__ == "__main__":
    main()
