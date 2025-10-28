#!/usr/bin/env python3
"""
Leaf Species Identifier
A system that first validates if an image is a leaf, then identifies the plant species.
Combines leaf validation and species classification.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import logging

# Import utilities
from preprocessing import ImagePreprocessor
from leaf_validator import LeafValidator
from feature_extraction import FeatureExtractor
from classification import PlantClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LeafSpeciesIdentifier:
    """Main class for leaf validation and species identification"""

    def __init__(self, validator_model_path=None, classifier_model_path=None):
        """
        Initialize the identifier

        Args:
            validator_model_path: Path to pre-trained leaf validator model
            classifier_model_path: Path to pre-trained species classifier model
        """
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()

        # Initialize models
        self.leaf_validator = LeafValidator()
        self.plant_classifier = PlantClassifier()

        # Load models if paths provided, with defaults
        validator_path = validator_model_path or 'models/leaf_rf.joblib'
        classifier_path = classifier_model_path or 'models/plant_classifier.pkl'

        if os.path.exists(validator_path):
            self.leaf_validator.load_model(validator_path)
            logger.info(f"Loaded leaf validator from {validator_path}")
        else:
            logger.warning("No leaf validator model provided or found")

        if os.path.exists(classifier_path):
            self.plant_classifier.load_model(classifier_path)
            logger.info(f"Loaded plant classifier from {classifier_path}")
        else:
            logger.warning("No plant classifier model provided or found")

    def process_image(self, image_path):
        """
        Process a single image: validate if leaf, then classify species

        Args:
            image_path: Path to the input image

        Returns:
            dict: Results containing validation and classification info
        """
        logger.info(f"Processing image: {image_path}")

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = {
            'image_path': image_path,
            'is_leaf': False,
            'leaf_confidence': 0.0,
            'species': None,
            'species_confidence': 0.0,
            'processing_successful': False
        }

        try:
            # Step 1: Validate if it's a leaf
            logger.info("Validating if image is a leaf...")
            validation_result = self.leaf_validator.validate_leaf(image)

            results['is_leaf'] = validation_result['is_leaf']
            results['leaf_confidence'] = validation_result['confidence']

            logger.info(f"Leaf validation: {results['is_leaf']} (confidence: {results['leaf_confidence']:.3f})")

        # Step 2: If it's a leaf, classify species
            if results['is_leaf']:
                logger.info("Image validated as leaf, proceeding to species classification...")

                # Preprocess for classification
                processed = self.preprocessor.resize_image(image)
                processed = self.preprocessor.to_grayscale(processed)

                # Extract features
                features = self.feature_extractor.extract_all_features(processed)
                feature_vector = np.array(list(features.values()))
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

                # Ensure feature vector matches classifier expectations (124 features)
                if len(feature_vector) > 124:
                    feature_vector = feature_vector[:124]
                elif len(feature_vector) < 124:
                    feature_vector = np.pad(feature_vector, (0, 124 - len(feature_vector)), 'constant')

                # Classify species
                if self.plant_classifier.is_trained:
                    classification_result = self.plant_classifier.predict_single(feature_vector)

                    results['species'] = classification_result['predicted_class']
                    results['species_confidence'] = classification_result['confidence']
                    results['top_predictions'] = classification_result.get('top_predictions', [])

                    logger.info(f"Species classification: {results['species']} "
                              f"(confidence: {results['species_confidence']:.3f})")
                else:
                    logger.warning("Plant classifier not trained, skipping species classification")
            else:
                logger.info("Image is not a leaf, skipping species classification")

            results['processing_successful'] = True

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            results['error'] = str(e)

        return results

    def process_directory(self, input_dir, output_file=None):
        """
        Process all images in a directory

        Args:
            input_dir: Directory containing images
            output_file: Optional file to save results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'**/*{ext}')))
            image_files.extend(list(input_path.glob(f'**/*{ext.upper()}')))

        logger.info(f"Found {len(image_files)} images in {input_dir}")

        all_results = []
        for img_file in image_files:
            try:
                result = self.process_image(str(img_file))
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")
                all_results.append({
                    'image_path': str(img_file),
                    'error': str(e),
                    'processing_successful': False
                })

        # Save results if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")

        return all_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Leaf Species Identifier')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--input-dir', type=str, help='Directory containing images to process')
    parser.add_argument('--output-file', type=str, help='File to save results (JSON)')
    parser.add_argument('--validator-model', type=str, default='models/leaf_validator.pkl',
                       help='Path to leaf validator model')
    parser.add_argument('--classifier-model', type=str, default='models/plant_classifier.pkl',
                       help='Path to plant classifier model')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.image and not args.input_dir:
        parser.error("Either --image or --input-dir must be specified")

    # Initialize identifier
    identifier = LeafSpeciesIdentifier(
        validator_model_path=args.validator_model,
        classifier_model_path=args.classifier_model
    )

    try:
        if args.image:
            # Process single image
            result = identifier.process_image(args.image)

            # Print results
            print("\n" + "="*50)
            print("LEAF SPECIES IDENTIFICATION RESULTS")
            print("="*50)
            print(f"Image: {result['image_path']}")
            print(f"Is Leaf: {result['is_leaf']}")
            print(f"Leaf Confidence: {result['leaf_confidence']:.3f}")

            if result['is_leaf'] and result['species']:
                print(f"Predicted Species: {result['species']}")
                print(f"Species Confidence: {result['species_confidence']:.3f}")

                if result.get('top_predictions'):
                    print("\nTop Predictions:")
                    for pred in result['top_predictions'][:5]:
                        print(f"  {pred['class']}: {pred['confidence']:.3f}")
            else:
                print("Species: Not classified (not a leaf)")

            if not result['processing_successful']:
                print(f"Error: {result.get('error', 'Unknown error')}")

            print("="*50)

        elif args.input_dir:
            # Process directory
            results = identifier.process_directory(args.input_dir, args.output_file)

            # Print summary
            total_images = len(results)
            leaf_images = sum(1 for r in results if r.get('is_leaf', False))
            classified_images = sum(1 for r in results if r.get('species') is not None)
            successful = sum(1 for r in results if r.get('processing_successful', False))

            print("\n" + "="*50)
            print("DIRECTORY PROCESSING SUMMARY")
            print("="*50)
            print(f"Total images processed: {total_images}")
            print(f"Images identified as leaves: {leaf_images}")
            print(f"Species classifications: {classified_images}")
            print(f"Successful processing: {successful}")
            print(".1f")
            print(".1f")
            print("="*50)

            if args.output_file:
                print(f"Detailed results saved to: {args.output_file}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
