"""
Leaf Validation Module for Plant Species Identification
Determines if an uploaded image contains a valid plant leaf
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import Dict, List, Tuple, Optional, Union
from skimage import feature, measure, morphology, filters
from scipy import ndimage
import matplotlib.pyplot as plt


class LeafValidator:
    """Validates if an image contains a plant leaf"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize leaf validator
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.model = None
        self.scaler = None
        self.feature_names = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using Local Binary Pattern and other methods
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        
        # Add LBP features
        for i, val in enumerate(lbp_hist):
            features[f'lbp_bin_{i}'] = val
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        try:
            glcm = feature.graycomatrix(image, distances=[1], angles=[0, 45, 90, 135], 
                                      levels=256, symmetric=True, normed=True)
            
            # GLCM properties
            contrast = feature.graycoprops(glcm, 'contrast').mean()
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
            energy = feature.graycoprops(glcm, 'energy').mean()
            correlation = feature.graycoprops(glcm, 'correlation').mean()
            
            features.update({
                'glcm_contrast': contrast,
                'glcm_dissimilarity': dissimilarity,
                'glcm_homogeneity': homogeneity,
                'glcm_energy': energy,
                'glcm_correlation': correlation
            })
        except Exception as e:
            print(f"GLCM calculation failed: {e}")
            features.update({
                'glcm_contrast': 0, 'glcm_dissimilarity': 0, 'glcm_homogeneity': 0,
                'glcm_energy': 0, 'glcm_correlation': 0
            })
        
        # Statistical texture features
        features.update({
            'texture_mean': np.mean(image),
            'texture_std': np.std(image),
            'texture_skewness': self._calculate_skewness(image),
            'texture_kurtosis': self._calculate_kurtosis(image),
            'texture_entropy': self._calculate_entropy(image)
        })
        
        return features
    
    def extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract shape-based features
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary of shape features
        """
        features = {}
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {f'shape_{i}': 0 for i in range(15)}  # Return zeros if no contours
        
        # Get the largest contour (assumed to be the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic shape properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        
        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        circle_area = np.pi * radius * radius
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate features
        features.update({
            'shape_area': area,
            'shape_perimeter': perimeter,
            'shape_aspect_ratio': w / h if h > 0 else 0,
            'shape_extent': area / rect_area if rect_area > 0 else 0,
            'shape_solidity': area / hull_area if hull_area > 0 else 0,
            'shape_compactness': (perimeter * perimeter) / area if area > 0 else 0,
            'shape_circularity': (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0,
            'shape_roundness': (4 * area) / (np.pi * max(w, h) * max(w, h)) if max(w, h) > 0 else 0,
            'shape_rectangularity': area / rect_area if rect_area > 0 else 0,
            'shape_convexity': perimeter / cv2.arcLength(hull, True) if cv2.arcLength(hull, True) > 0 else 0,
            'shape_eccentricity': self._calculate_eccentricity(largest_contour),
            'shape_elongation': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
            'shape_orientation': self._calculate_orientation(largest_contour),
            'shape_hu_moments_mean': np.mean(cv2.HuMoments(cv2.moments(largest_contour)).flatten()),
            'shape_hu_moments_std': np.std(cv2.HuMoments(cv2.moments(largest_contour)).flatten())
        })
        
        return features
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-based features (for RGB images)
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Dictionary of color features
        """
        features = {}
        
        if len(image.shape) == 3:
            # RGB image
            for i, channel in enumerate(['red', 'green', 'blue']):
                ch = image[:, :, i]
                features.update({
                    f'color_{channel}_mean': np.mean(ch),
                    f'color_{channel}_std': np.std(ch),
                    f'color_{channel}_skewness': self._calculate_skewness(ch),
                    f'color_{channel}_kurtosis': self._calculate_kurtosis(ch)
                })
            
            # Color ratios
            r_mean, g_mean, b_mean = [np.mean(image[:, :, i]) for i in range(3)]
            total = r_mean + g_mean + b_mean
            if total > 0:
                features.update({
                    'color_r_ratio': r_mean / total,
                    'color_g_ratio': g_mean / total,
                    'color_b_ratio': b_mean / total
                })
            else:
                features.update({'color_r_ratio': 0, 'color_g_ratio': 0, 'color_b_ratio': 0})
        else:
            # Grayscale image
            features.update({
                'color_gray_mean': np.mean(image),
                'color_gray_std': np.std(image),
                'color_gray_skewness': self._calculate_skewness(image),
                'color_gray_kurtosis': self._calculate_kurtosis(image),
                'color_r_ratio': 0, 'color_g_ratio': 0, 'color_b_ratio': 0
            })
        
        return features
    
    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract edge-based features
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary of edge features
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Edge direction histogram
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Quantize orientations
        hist, _ = np.histogram(orientation.flatten(), bins=8, range=(-np.pi, np.pi))
        hist = hist.astype(float) / np.sum(hist)
        
        features = {
            'edge_density': edge_density,
            'edge_total_pixels': edge_pixels,
            'edge_strength_mean': np.mean(edges[edges > 0]) if edge_pixels > 0 else 0,
            'edge_strength_std': np.std(edges[edges > 0]) if edge_pixels > 0 else 0
        }
        
        # Add orientation histogram features
        for i, val in enumerate(hist):
            features[f'edge_orientation_{i}'] = val
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all features from an image
        
        Args:
            image: Input image
            
        Returns:
            Feature vector as numpy array
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Extract all feature types
        texture_features = self.extract_texture_features(gray_image)
        shape_features = self.extract_shape_features(gray_image)
        color_features = self.extract_color_features(image)
        edge_features = self.extract_edge_features(gray_image)
        
        # Combine all features
        all_features = {**texture_features, **shape_features, **color_features, **edge_features}
        
        # Store feature names for later use
        if not self.feature_names:
            self.feature_names = list(all_features.keys())
        
        # Convert to numpy array
        feature_vector = np.array([all_features.get(name, 0) for name in self.feature_names])
        
        return feature_vector
    
    def train_validator(self, leaf_images: List[np.ndarray], non_leaf_images: List[np.ndarray],
                       model_type: str = 'random_forest', save_path: Optional[str] = None) -> Dict:
        """
        Train the leaf validator
        
        Args:
            leaf_images: List of leaf images
            non_leaf_images: List of non-leaf images
            model_type: 'random_forest' or 'svm'
            save_path: Path to save trained model
            
        Returns:
            Training results dictionary
        """
        print("Extracting features from training images...")
        
        # Extract features
        X = []
        y = []
        
        # Process leaf images
        for img in leaf_images:
            features = self.extract_all_features(img)
            X.append(features)
            y.append(1)  # Leaf class
        
        # Process non-leaf images
        for img in non_leaf_images:
            features = self.extract_all_features(img)
            X.append(features)
            y.append(0)  # Non-leaf class
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError("model_type must be 'random_forest' or 'svm'")
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed. Accuracy: {accuracy:.3f}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self._get_feature_importance(),
            'n_features': len(self.feature_names)
        }
        
        return results
    
    def predict(self, image: np.ndarray) -> Dict[str, Union[bool, float]]:
        """
        Predict if an image contains a leaf
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded. Train model first or load pre-trained model.")
        
        # Extract features
        features = self.extract_all_features(image)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'is_leaf': bool(prediction),
            'confidence': float(probability[prediction]),
            'leaf_probability': float(probability[1]),
            'non_leaf_probability': float(probability[0])
        }
    
    def validate_leaf(self, image: np.ndarray) -> Dict[str, Union[bool, float]]:
        """
        Validate if an image contains a leaf

        Args:
            image: Input image

        Returns:
            Dictionary with validation results
        """
        # Try ML-based prediction first if model is available
        if self.model is not None and self.scaler is not None:
            try:
                return self.predict(image)
            except Exception as e:
                print(f"ML prediction failed, falling back to rule-based: {e}")

        # Fall back to rule-based validation
        return self.validate_with_rules(image)

    def validate_with_rules(self, image: np.ndarray) -> Dict[str, Union[bool, float, str]]:
        """
        Rule-based leaf validation (backup method)

        Args:
            image: Input image

        Returns:
            Dictionary with validation results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()

        # Extract basic features
        shape_features = self.extract_shape_features(gray_image)
        edge_features = self.extract_edge_features(gray_image)
        color_features = self.extract_color_features(image)

        # Rule-based validation
        reasons = []
        score = 0

        # Check aspect ratio (leaves are usually not extremely elongated)
        aspect_ratio = shape_features.get('shape_aspect_ratio', 0)
        if 0.3 <= aspect_ratio <= 3.0:
            score += 1
            reasons.append("Good aspect ratio")

        # Check solidity (leaves should have reasonable solidity)
        solidity = shape_features.get('shape_solidity', 0)
        if solidity > 0.7:
            score += 1
            reasons.append("Good solidity")

        # Check edge density (leaves should have moderate edge density)
        edge_density = edge_features.get('edge_density', 0)
        if 0.05 <= edge_density <= 0.3:
            score += 1
            reasons.append("Good edge density")

        # Check circularity (leaves shouldn't be too circular)
        circularity = shape_features.get('shape_circularity', 0)
        if circularity < 0.8:
            score += 1
            reasons.append("Not too circular")

        # Check if image has green tint (for color images)
        if len(image.shape) == 3:
            g_ratio = color_features.get('color_g_ratio', 0)
            if g_ratio > 0.35:  # Green dominant
                score += 1
                reasons.append("Green dominant")

        # Final decision
        is_leaf = score >= 3
        confidence = score / 5.0

        return {
            'is_leaf': is_leaf,
            'confidence': confidence,
            'score': score,
            'max_score': 5,
            'reasons': reasons
        }
    
    def save_model(self, save_path: str):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load pre-trained model and scaler"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {model_path}")
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_eccentricity(self, contour: np.ndarray) -> float:
        """Calculate eccentricity of contour"""
        try:
            ellipse = cv2.fitEllipse(contour)
            a, b = ellipse[1]  # Major and minor axes
            if a == 0:
                return 0
            eccentricity = np.sqrt(1 - (min(a, b) / max(a, b)) ** 2)
            return eccentricity
        except:
            return 0
    
    def _calculate_orientation(self, contour: np.ndarray) -> float:
        """Calculate orientation of contour"""
        try:
            ellipse = cv2.fitEllipse(contour)
            return ellipse[2]  # Angle
        except:
            return 0
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {}


def create_synthetic_training_data(n_leaf_samples: int = 100, n_non_leaf_samples: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create synthetic training data for demonstration
    
    Args:
        n_leaf_samples: Number of synthetic leaf samples
        n_non_leaf_samples: Number of synthetic non-leaf samples
        
    Returns:
        Tuple of (leaf_images, non_leaf_images)
    """
    leaf_images = []
    non_leaf_images = []
    
    # Create synthetic leaf-like images
    for _ in range(n_leaf_samples):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Create leaf-like shape
        center = (112, 112)
        axes = (np.random.randint(40, 80), np.random.randint(60, 100))
        angle = np.random.randint(0, 180)
        
        # Draw ellipse (leaf shape)
        cv2.ellipse(img, center, axes, angle, 0, 360, (0, 150, 0), -1)
        
        # Add some texture
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        leaf_images.append(img)
    
    # Create synthetic non-leaf images
    for _ in range(n_non_leaf_samples):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add geometric shapes (rectangles, circles)
        if np.random.random() > 0.5:
            cv2.rectangle(img, (50, 50), (170, 170), (np.random.randint(0, 255),) * 3, -1)
        else:
            cv2.circle(img, (112, 112), np.random.randint(30, 70), (np.random.randint(0, 255),) * 3, -1)
        
        non_leaf_images.append(img)
    
    return leaf_images, non_leaf_images


if __name__ == "__main__":
    # Example usage
    validator = LeafValidator()
    
    # Create synthetic training data for demonstration
    print("Creating synthetic training data...")
    leaf_images, non_leaf_images = create_synthetic_training_data(50, 50)
    
    # Train validator
    results = validator.train_validator(leaf_images, non_leaf_images, model_type='random_forest')
    print("Training Results:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Number of features: {results['n_features']}")
    
    # Test prediction on a sample image
    test_image = leaf_images[0]
    prediction = validator.predict(test_image)
    print(f"\nPrediction: {prediction}")
    
    # Test rule-based validation
    rule_result = validator.validate_with_rules(test_image)
    print(f"Rule-based validation: {rule_result}")