#!/usr/bin/env python3
"""
Test script for the plant species identification pipeline
Tests with various sample images and validates performance
"""

import os
import glob
from pathlib import Path
from plant_species_id.main_pipeline import identify_plant_species
import json
from datetime import datetime

def test_pipeline():
    """Test the pipeline with available images"""

    # Get all image files in current directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(ext))

    # Filter out any non-sample images (like those in data/ or results/)
    sample_images = [f for f in image_files if not any(x in f for x in ['data/', 'results/', 'models/'])]

    print(f"Found {len(sample_images)} sample images to test:")
    for img in sample_images:
        print(f"  - {img}")

    print("\n" + "="*60)
    print("TESTING PLANT SPECIES IDENTIFICATION PIPELINE")
    print("="*60)

    results = []
    leaf_count = 0
    species_counts = {}

    for i, image_path in enumerate(sample_images, 1):
        print(f"\n[{i}/{len(sample_images)}] Testing: {image_path}")
        print("-" * 40)

        try:
            result = identify_plant_species(image_path, debug=False)

            # Store result
            result['image_path'] = image_path
            results.append(result)

            # Print summary
            print(f"Success: {result['success']}")
            print(f"Is Leaf: {result['is_leaf']}")

            if result['is_leaf']:
                leaf_count += 1
                species = result['species']
                confidence = result['confidence']
                print(f"Species: {species}")
                print(f"Confidence: {confidence:.1%}")

                # Count species
                if species in species_counts:
                    species_counts[species] += 1
                else:
                    species_counts[species] = 1

                print(f"Message: {result['message']}")
            else:
                print(f"Message: {result['message']}")

            if not result['success']:
                print(f"Error: {result['error']}")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'is_leaf': False,
                'species': None,
                'confidence': 0.0
            })

    # Summary statistics
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_images = len(results)
    successful_identifications = sum(1 for r in results if r['success'])
    leaf_identifications = sum(1 for r in results if r.get('is_leaf', False))

    print(f"Total images tested: {total_images}")
    print(f"Successful processing: {successful_identifications}/{total_images} ({successful_identifications/total_images*100:.1f}%)")
    print(f"Leaf images detected: {leaf_identifications}/{total_images} ({leaf_identifications/total_images*100:.1f}%)")
    print(f"Non-leaf images rejected: {total_images - leaf_identifications}/{total_images} ({(total_images - leaf_identifications)/total_images*100:.1f}%)")

    if species_counts:
        print(f"\nSpecies distribution:")
        for species, count in sorted(species_counts.items()):
            print(f"  {species}: {count} images")

    # Average confidence for leaf identifications
    leaf_confidences = [r['confidence'] for r in results if r.get('is_leaf', False)]
    if leaf_confidences:
        avg_confidence = sum(leaf_confidences) / len(leaf_confidences)
        print(".1%")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/test_results_{timestamp}.json"

    os.makedirs("results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    # Print failed cases
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\nFailed cases ({len(failed_results)}):")
        for r in failed_results:
            print(f"  - {r['image_path']}: {r.get('error', 'Unknown error')}")

    return results

def test_specific_cases():
    """Test specific edge cases"""

    print("\n" + "="*60)
    print("TESTING SPECIFIC EDGE CASES")
    print("="*60)

    test_cases = [
        ("greenScreen.png", "Green screen (should be rejected)"),
        ("greenScreen2.png", "Another green screen (should be rejected)"),
        ("Green-Screen-PNG.png", "Green screen PNG (should be rejected)"),
        ("a.jpg", "Known leaf image (should be accepted)"),
        ("AloveraLeaf.jpg", "Aloe vera leaf (should be accepted)"),
    ]

    for image_path, description in test_cases:
        if os.path.exists(image_path):
            print(f"\nTesting: {image_path}")
            print(f"Description: {description}")

            result = identify_plant_species(image_path, debug=False)

            print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
            print(f"Is Leaf: {result['is_leaf']}")

            if result['is_leaf']:
                print(f"Species: {result['species']} ({result['confidence']:.1%})")
            else:
                print("Correctly rejected as non-leaf")

            if not result['success']:
                print(f"Error: {result['error']}")
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    # Run comprehensive tests
    test_pipeline()

    # Run specific edge case tests
    test_specific_cases()

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
