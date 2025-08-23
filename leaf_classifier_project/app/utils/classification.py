"""
Classification Module for Plant Species Identification
Implements various classification methods including traditional ML and deep learning
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, applications, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Deep learning features will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Some deep learning features will be disabled.")


class PlantClassifier:
    """Main classifier for plant species identification"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier
        
        Args:
            model_type: Type of model ('knn', 'svm', 'random_forest', 'gradient_boost', 
                       'logistic', 'cnn', 'mobilenet', 'resnet')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.class_names = []
        self.is_trained = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on type"""
        if self.model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type in ['cnn', 'mobilenet', 'resnet']:
            if not TENSORFLOW_AVAILABLE:
                raise ValueError("TensorFlow is required for deep learning models")
            self.model = None  # Will be created during training
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize scaler and label encoder
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def train_traditional_ml(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: Optional[List[str]] = None,
                           class_names: Optional[List[str]] = None,
                           optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train traditional ML models
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            class_names: Names of classes
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        print(f"Training {self.model_type} classifier...")
        
        # Store feature and class names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = class_names or self.label_encoder.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            self.model = self._optimize_hyperparameters(X_train_scaled, y_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled) if hasattr(self.model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.class_names),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self._get_feature_importance(),
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_labels': y_test
        }
        
        print(f"Training completed. Accuracy: {accuracy:.3f} (±{cv_scores.std():.3f})")
        
        return results
    
    def train_deep_learning(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                          input_shape: Tuple[int, int, int] = (224, 224, 3),
                          epochs: int = 50, batch_size: int = 32,
                          learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train deep learning models
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            input_shape: Input image shape
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training results dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow is required for deep learning models")
        
        print(f"Training {self.model_type} deep learning model...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_
        
        # Convert to categorical
        y_train_cat = keras.utils.to_categorical(y_train_encoded, n_classes)
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            y_val_cat = keras.utils.to_categorical(y_val_encoded, n_classes)
            validation_data = (X_val, y_val_cat)
        else:
            # Split training data for validation
            X_train, X_val, y_train_cat, y_val_cat = train_test_split(
                X_train, y_train_cat, test_size=0.2, random_state=42
            )
            validation_data = (X_val, y_val_cat)
        
        # Create model
        if self.model_type == 'cnn':
            self.model = self._create_cnn_model(input_shape, n_classes)
        elif self.model_type == 'mobilenet':
            self.model = self._create_mobilenet_model(input_shape, n_classes)
        elif self.model_type == 'resnet':
            self.model = self._create_resnet_model(input_shape, n_classes)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val_cat, verbose=0)
        
        self.is_trained = True
        
        results = {
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'history': history.history,
            'n_classes': n_classes,
            'class_names': self.class_names,
            'model_summary': self._get_model_summary()
        }
        
        print(f"Training completed. Validation accuracy: {val_accuracy:.3f}")
        
        return results
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Input features or images
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Predictions (and probabilities if requested)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type in ['cnn', 'mobilenet', 'resnet']:
            # Deep learning prediction
            probabilities = self.model.predict(X)
            predictions = np.argmax(probabilities, axis=1)
            
            # Convert back to original labels
            predictions = self.label_encoder.inverse_transform(predictions)
            
            if return_probabilities:
                return predictions, probabilities
            return predictions
        
        else:
            # Traditional ML prediction
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Convert back to original labels
            predictions = self.label_encoder.inverse_transform(predictions)
            
            if return_probabilities:
                probabilities = self.model.predict_proba(X_scaled)
                return predictions, probabilities
            return predictions
    
    def predict_single(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Predict single sample with detailed results
        
        Args:
            x: Single sample (feature vector or image)
            
        Returns:
            Dictionary with prediction details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure x is 2D for traditional ML or 4D for deep learning
        if self.model_type in ['cnn', 'mobilenet', 'resnet']:
            if x.ndim == 3:
                x = np.expand_dims(x, axis=0)
            probabilities = self.model.predict(x)[0]
            prediction_idx = np.argmax(probabilities)
        else:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_scaled = self.scaler.transform(x)
            probabilities = self.model.predict_proba(x_scaled)[0]
            prediction_idx = np.argmax(probabilities)
        
        predicted_class = self.label_encoder.inverse_transform([prediction_idx])[0]
        confidence = probabilities[prediction_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_predictions.append({
                'class': class_name,
                'probability': float(prob),
                'confidence': float(prob * 100)
            })
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'confidence_percent': float(confidence * 100),
            'top_predictions': top_predictions,
            'all_probabilities': {
                self.label_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features/images
            y_test: Test labels
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions, probabilities = self.predict(X_test, return_probabilities=True)
        
        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)
        pred_encoded = self.label_encoder.transform(predictions)
        
        accuracy = accuracy_score(y_test_encoded, pred_encoded)
        conf_matrix = confusion_matrix(y_test_encoded, pred_encoded)
        class_report = classification_report(y_test_encoded, pred_encoded, 
                                          target_names=self.class_names)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': y_test
        }
    
    def save_model(self, save_path: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if self.model_type in ['cnn', 'mobilenet', 'resnet']:
            # Save deep learning model
            self.model.save(f"{save_path}_model.h5")
            
            # Save additional components
            model_data = {
                'model_type': self.model_type,
                'label_encoder': self.label_encoder,
                'class_names': self.class_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, f"{save_path}_components.pkl")
        else:
            # Save traditional ML model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load trained model"""
        if self.model_type in ['cnn', 'mobilenet', 'resnet']:
            # Load deep learning model
            self.model = keras.models.load_model(f"{load_path}_model.h5")
            
            # Load additional components
            model_data = joblib.load(f"{load_path}_components.pkl")
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            self.is_trained = model_data['is_trained']
        else:
            # Load traditional ML model
            model_data = joblib.load(load_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {load_path}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        y_true_encoded = self.label_encoder.transform(y_true)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot training history for deep learning models"""
        if self.model_type not in ['cnn', 'mobilenet', 'resnet']:
            print("Training history plotting is only available for deep learning models")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Optimize hyperparameters using GridSearchCV"""
        print("Optimizing hyperparameters...")
        
        param_grids = {
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            }
        }
        
        if self.model_type in param_grids:
            grid_search = GridSearchCV(
                self.model, param_grids[self.model_type],
                cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X, y)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return self.model
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def _create_cnn_model(self, input_shape: Tuple[int, int, int], n_classes: int):
        """Create custom CNN model"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation='softmax')
        ])
        return model
    
    def _create_mobilenet_model(self, input_shape: Tuple[int, int, int], n_classes: int):
        """Create MobileNetV2-based model"""
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self, input_shape: Tuple[int, int, int], n_classes: int):
        """Create ResNet50-based model"""
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        return model
    
    def _get_model_summary(self) -> str:
        """Get model summary as string"""
        if self.model_type in ['cnn', 'mobilenet', 'resnet']:
            import io
            import sys
            
            # Capture model summary
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            
            return buffer.getvalue()
        else:
            return str(self.model)


class HybridClassifier:
    """Hybrid classifier combining rule-based and ML approaches"""
    
    def __init__(self, ml_classifier: PlantClassifier, leaf_validator=None):
        """
        Initialize hybrid classifier
        
        Args:
            ml_classifier: Trained ML classifier
            leaf_validator: Leaf validator for pre-filtering
        """
        self.ml_classifier = ml_classifier
        self.leaf_validator = leaf_validator
        self.confidence_threshold = 0.7
    
    def predict(self, image: np.ndarray, features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Hybrid prediction combining validation and classification
        
        Args:
            image: Input image
            features: Extracted features (if available)
            
        Returns:
            Prediction results with validation
        """
        results = {
            'is_valid_leaf': True,
            'leaf_confidence': 1.0,
            'predicted_class': None,
            'classification_confidence': 0.0,
            'method_used': 'hybrid',
            'validation_details': None
        }
        
        # Step 1: Leaf validation
        if self.leaf_validator is not None:
            validation_result = self.leaf_validator.predict(image)
            results['is_valid_leaf'] = validation_result['is_leaf']
            results['leaf_confidence'] = validation_result['confidence']
            results['validation_details'] = validation_result
            
            if not validation_result['is_leaf']:
                results['predicted_class'] = 'not_a_leaf'
                results['classification_confidence'] = validation_result['confidence']
                return results
        
        # Step 2: Species classification
        if features is not None:
            # Use features for traditional ML
            prediction_result = self.ml_classifier.predict_single(features)
        else:
            # Use image for deep learning
            prediction_result = self.ml_classifier.predict_single(image)
        
        results['predicted_class'] = prediction_result['predicted_class']
        results['classification_confidence'] = prediction_result['confidence']
        results['top_predictions'] = prediction_result['top_predictions']
        
        # Step 3: Apply confidence thresholding
        if prediction_result['confidence'] < self.confidence_threshold:
            results['predicted_class'] = 'unknown_species'
            results['method_used'] = 'low_confidence_rejection'
        
        return results
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for predictions"""
        self.confidence_threshold = threshold


if __name__ == "__main__":
    # Example usage
    print("Plant Classification Module")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    n_classes = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([f'species_{i}' for i in range(n_classes)], n_samples)
    
    # Test traditional ML classifier
    classifier = PlantClassifier(model_type='random_forest')
    results = classifier.train_traditional_ml(X, y)
    
    print(f"Training completed with accuracy: {results['accuracy']:.3f}")
    
    # Test prediction
    test_sample = X[0]
    prediction = classifier.predict_single(test_sample)
    print(f"Sample prediction: {prediction['predicted_class']} (confidence: {prediction['confidence_percent']:.1f}%)")