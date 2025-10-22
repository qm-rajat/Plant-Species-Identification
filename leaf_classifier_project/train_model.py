#!/usr/bin/env python3
"""
Training Pipeline for Plant Species Identification System
Comprehensive training script for leaf classification models
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import cv2
from tqdm import tqdm
import logging

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.preprocessing import ImagePreprocessor
from app.utils.edge_detection import EdgeDetector
from app.utils.feature_extraction import FeatureExtractor
from app.utils.leaf_validator import LeafValidator, create_synthetic_training_data
from app.utils.classification import PlantClassifier
from app.utils.evaluation import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for plant species identification"""
    
    def __init__(self, config: dict):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.edge_detector = EdgeDetector()
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()
        
        # Initialize models
        self.leaf_validator = LeafValidator()
        self.plant_classifier = None
        
        # Data storage
        self.training_data = {
            'images': [],
            'labels': [],
            'features': [],
            'file_paths': []
        }
        
        self.validation_data = {
            'leaf_images': [],
            'non_leaf_images': []
        }
        
        # Results storage
        self.results = {
            'leaf_validator': {},
            'plant_classifier': {},
            'edge_detection': {},
            'feature_extraction': {},
            'evaluation': {}
        }
    
    def load_dataset(self, data_dir: str):
        """
        Load dataset from directory structure
        
        Args:
            data_dir: Root directory containing organized dataset
                     Expected structure:
                     data_dir/
                     ├── species_1/
                     │   ├── image1.jpg
                     │   └── image2.jpg
                     ├── species_2/
                     └── non_leaf/ (optional)
        """
        logger.info(f"Loading dataset from {data_dir}")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # Get all species directories
        species_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for species_dir in tqdm(species_dirs, desc="Loading species"):
            species_name = species_dir.name
            
            # Handle non-leaf images separately
            if species_name.lower() == 'non_leaf':
                self._load_non_leaf_images(species_dir)
                continue
            
            # Load species images
            image_files = list(species_dir.glob('*.jpg')) + \
                         list(species_dir.glob('*.png')) + \
                         list(species_dir.glob('*.jpeg'))
            
            logger.info(f"Loading {len(image_files)} images for {species_name}")
            
            for img_file in image_files:
                try:
                    # Load and preprocess image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Store data
                    self.training_data['images'].append(image)
                    self.training_data['labels'].append(species_name)
                    self.training_data['file_paths'].append(str(img_file))
                    
                    # Add to leaf validation data
                    self.validation_data['leaf_images'].append(image)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
        
        logger.info(f"Loaded {len(self.training_data['images'])} images from "
                   f"{len(set(self.training_data['labels']))} species")
    
    def _load_non_leaf_images(self, non_leaf_dir: Path):
        """Load non-leaf images for validation training"""
        image_files = list(non_leaf_dir.glob('*.jpg')) + \
                     list(non_leaf_dir.glob('*.png')) + \
                     list(non_leaf_dir.glob('*.jpeg'))
        
        logger.info(f"Loading {len(image_files)} non-leaf images")
        
        for img_file in image_files:
            try:
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.validation_data['non_leaf_images'].append(image)
                
            except Exception as e:
                logger.warning(f"Failed to load non-leaf image {img_file}: {e}")
    
    def create_synthetic_data(self):
        """Create synthetic training data if real data is not available"""
        logger.info("Creating synthetic training data")
        
        # Create synthetic leaf validation data
        leaf_images, non_leaf_images = create_synthetic_training_data(
            n_leaf_samples=self.config.get('synthetic_leaf_samples', 200),
            n_non_leaf_samples=self.config.get('synthetic_non_leaf_samples', 200)
        )
        
        self.validation_data['leaf_images'].extend(leaf_images)
        self.validation_data['non_leaf_images'].extend(non_leaf_images)
        
        # Create synthetic species data
        species_names = [f'Species_{i}' for i in range(self.config.get('n_synthetic_species', 5))]
        samples_per_species = self.config.get('samples_per_species', 50)
        
        for species in species_names:
            for _ in range(samples_per_species):
                # Create synthetic leaf image with species-specific characteristics
                img = self._create_synthetic_species_image(species)
                
                self.training_data['images'].append(img)
                self.training_data['labels'].append(species)
                self.training_data['file_paths'].append(f'synthetic_{species}_{len(self.training_data["images"])}.png')
        
        logger.info(f"Created {len(self.training_data['images'])} synthetic training samples")
    
    def _create_synthetic_species_image(self, species_name: str) -> np.ndarray:
        """Create a synthetic image for a specific species"""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Create species-specific characteristics
        species_id = hash(species_name) % 1000
        np.random.seed(species_id)
        
        # Base leaf shape
        center = (112, 112)
        axes = (np.random.randint(50, 90), np.random.randint(70, 110))
        angle = np.random.randint(0, 180)
        
        # Species-specific color variations
        base_color = (
            np.random.randint(20, 80),    # Red
            np.random.randint(100, 200),  # Green (dominant)
            np.random.randint(20, 80)     # Blue
        )
        
        # Draw main leaf shape
        cv2.ellipse(img, center, axes, angle, 0, 360, base_color, -1)
        
        # Add species-specific patterns
        if 'Species_0' in species_name:
            # Add circular patterns
            for _ in range(5):
                center_pt = (np.random.randint(50, 174), np.random.randint(50, 174))
                cv2.circle(img, center_pt, np.random.randint(5, 15), 
                          (base_color[0]+20, base_color[1]-20, base_color[2]+10), -1)
        
        elif 'Species_1' in species_name:
            # Add line patterns
            for _ in range(3):
                pt1 = (np.random.randint(0, 224), np.random.randint(0, 224))
                pt2 = (np.random.randint(0, 224), np.random.randint(0, 224))
                cv2.line(img, pt1, pt2, (base_color[0]-10, base_color[1]+10, base_color[2]), 2)
        
        # Add noise for realism
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def extract_features(self):
        """Extract features from all training images"""
        logger.info("Extracting features from training images")
        
        features_list = []
        
        for img in tqdm(self.training_data['images'], desc="Feature extraction"):
            try:
                # Extract comprehensive features
                features = self.feature_extractor.extract_all_features(img)
                feature_vector = np.array(list(features.values()))
                
                # Handle NaN values
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                
                features_list.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Feature extraction failed for image: {e}")
                # Use zero vector as fallback
                features_list.append(np.zeros(100))  # Placeholder size
        
        self.training_data['features'] = np.array(features_list)
        logger.info(f"Extracted features with shape: {self.training_data['features'].shape}")
    
    def train_leaf_validator(self):
        """Train the leaf validation model"""
        logger.info("Training leaf validator")
        
        if not self.validation_data['leaf_images'] or not self.validation_data['non_leaf_images']:
            logger.warning("No validation data available, creating synthetic data")
            leaf_images, non_leaf_images = create_synthetic_training_data(100, 100)
            self.validation_data['leaf_images'] = leaf_images
            self.validation_data['non_leaf_images'] = non_leaf_images
        
        # Train validator
        results = self.leaf_validator.train_validator(
            self.validation_data['leaf_images'],
            self.validation_data['non_leaf_images'],
            model_type=self.config.get('leaf_validator_model', 'random_forest')
        )
        
        self.results['leaf_validator'] = results
        
        # Save model
        validator_path = os.path.join(self.config['output_dir'], 'leaf_validator.pkl')
        self.leaf_validator.save_model(validator_path)
        
        logger.info(f"Leaf validator trained with accuracy: {results['accuracy']:.3f}")
    
    def train_plant_classifier(self):
        """Train the plant species classifier"""
        logger.info("Training plant species classifier")
        
        if len(self.training_data['features']) == 0:
            logger.error("No features available for training")
            return
        
        # Initialize classifier
        model_type = self.config.get('classifier_model', 'random_forest')
        self.plant_classifier = PlantClassifier(model_type=model_type)
        
        # Prepare data
        X = self.training_data['features']
        y = np.array(self.training_data['labels'])
        
        # Get unique class names
        class_names = list(set(y))
        
        # Train classifier
        if model_type in ['cnn', 'mobilenet', 'resnet']:
            # For deep learning models, use images directly
            images = np.array(self.training_data['images'])
            results = self.plant_classifier.train_deep_learning(
                images, y,
                epochs=self.config.get('epochs', 50),
                batch_size=self.config.get('batch_size', 32)
            )
        else:
            # For traditional ML models, use extracted features
            results = self.plant_classifier.train_traditional_ml(
                X, y,
                feature_names=[f'feature_{i}' for i in range(X.shape[1])],
                class_names=class_names,
                optimize_hyperparameters=self.config.get('optimize_hyperparameters', True)
            )
        
        self.results['plant_classifier'] = results
        
        # Save model
        classifier_path = os.path.join(self.config['output_dir'], 'plant_classifier.pkl')
        self.plant_classifier.save_model(classifier_path)
        
        logger.info(f"Plant classifier trained with accuracy: {results.get('accuracy', results.get('val_accuracy', 0)):.3f}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("Evaluating models")
        
        # Evaluate edge detection methods
        if len(self.training_data['images']) > 0:
            sample_images = self.training_data['images'][:10]  # Use first 10 images
            
            # Apply edge detection
            edge_results = {}
            for i, img in enumerate(sample_images):
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img
                
                img_results = self.edge_detector.apply_all_methods(gray_img)
                
                for method, result in img_results.items():
                    if method not in edge_results:
                        edge_results[method] = []
                    edge_results[method].append(result)
            
            # Evaluate edge detection
            edge_eval = self.evaluator.evaluate_edge_detection(
                sample_images, edge_results,
                save_dir=os.path.join(self.config['output_dir'], 'edge_evaluation')
            )
            
            self.results['edge_detection'] = edge_eval
        
        # Evaluate classification if we have test data
        if self.plant_classifier and self.plant_classifier.is_trained:
            # Create test split
            X = self.training_data['features']
            y = np.array(self.training_data['labels'])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Evaluate classifier
            eval_results = self.plant_classifier.evaluate(X_test, y_test)
            
            # Use evaluator for comprehensive analysis
            classification_eval = self.evaluator.evaluate_classification(
                y_test, eval_results['predictions'],
                eval_results.get('probabilities'),
                class_names=list(set(y)),
                model_name=self.plant_classifier.model_type,
                save_dir=os.path.join(self.config['output_dir'], 'classification_evaluation')
            )
            
            self.results['evaluation'] = classification_eval
    
    def save_results(self):
        """Save all training results"""
        logger.info("Saving training results")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Save comprehensive results
        results_file = os.path.join(self.config['output_dir'], 'training_results.json')
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            try:
                json_results[key] = self._make_json_serializable(value)
            except Exception as e:
                logger.warning(f"Could not serialize {key}: {e}")
                json_results[key] = str(value)
        
        # Add configuration and metadata
        json_results['config'] = self.config
        json_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        json_results['dataset_info'] = {
            'n_images': len(self.training_data['images']),
            'n_species': len(set(self.training_data['labels'])),
            'species_list': list(set(self.training_data['labels'])),
            'feature_dim': self.training_data['features'].shape[1] if len(self.training_data['features']) > 0 else 0
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate evaluation report
        report_file = os.path.join(self.config['output_dir'], 'evaluation_report.json')
        self.evaluator.generate_evaluation_report(report_file)
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other objects to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting complete training pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Load or create data
            if self.config.get('data_dir'):
                self.load_dataset(self.config['data_dir'])
            else:
                self.create_synthetic_data()
            
            # Step 2: Extract features
            self.extract_features()
            
            # Step 3: Train leaf validator
            self.train_leaf_validator()
            
            # Step 4: Train plant classifier
            self.train_plant_classifier()
            
            # Step 5: Evaluate models
            self.evaluate_models()
            
            # Step 6: Save results
            self.save_results()
            
            total_time = time.time() - start_time
            logger.info(f"Training pipeline completed successfully in {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Plant Species Identification Models')
    
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    # Model configuration
    parser.add_argument('--classifier-model', type=str, default='random_forest',
                       choices=['knn', 'svm', 'random_forest', 'gradient_boost', 'logistic', 'cnn', 'mobilenet', 'resnet'],
                       help='Type of classifier model')
    parser.add_argument('--leaf-validator-model', type=str, default='random_forest',
                       choices=['random_forest', 'svm'],
                       help='Type of leaf validator model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for deep learning models')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for deep learning models')
    parser.add_argument('--optimize-hyperparameters', action='store_true',
                       help='Enable hyperparameter optimization')
    
    # Synthetic data parameters
    parser.add_argument('--synthetic-leaf-samples', type=int, default=200,
                       help='Number of synthetic leaf samples')
    parser.add_argument('--synthetic-non-leaf-samples', type=int, default=200,
                       help='Number of synthetic non-leaf samples')
    parser.add_argument('--n-synthetic-species', type=int, default=5,
                       help='Number of synthetic species')
    parser.add_argument('--samples-per-species', type=int, default=50,
                       help='Number of samples per synthetic species')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'classifier_model': args.classifier_model,
        'leaf_validator_model': args.leaf_validator_model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'optimize_hyperparameters': args.optimize_hyperparameters,
        'synthetic_leaf_samples': args.synthetic_leaf_samples,
        'synthetic_non_leaf_samples': args.synthetic_non_leaf_samples,
        'n_synthetic_species': args.n_synthetic_species,
        'samples_per_species': args.samples_per_species
    })
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(config)
    success = pipeline.run_complete_pipeline()
    
    if success:
        logger.info("Training completed successfully!")
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Models saved to: {config['output_dir']}")
        print(f"Training logs: training.log")
        print(f"Results: {config['output_dir']}/training_results.json")
        print("="*50)
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()