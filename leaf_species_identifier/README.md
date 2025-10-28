# Leaf Species Identifier

A comprehensive plant species identification system with both command-line and web interfaces that uses machine learning and computer vision to identify plant species from leaf images.

## Features

- **Leaf Validation**: Advanced algorithms to verify if an image contains a leaf
- **Species Classification**: Machine learning models for accurate plant species identification
- **Feature Extraction**: Comprehensive feature extraction including geometric, vein, margin, texture, and color features
- **Web Interface**: User-friendly web application for easy image upload and results visualization
- **Command Line Interface**: CLI for batch processing and automation
- **Confidence Scores**: Detailed confidence scores and top predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd leaf_species_identifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

1. Start the Streamlit web server:
```bash
streamlit run streamlit_app.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Upload a leaf image and get instant identification results

### Command Line Interface

```bash
# Basic usage
python main.py --image path/to/leaf/image.jpg

# With custom models
python main.py --image path/to/leaf/image.jpg --validator-model path/to/validator.pkl --classifier-model path/to/classifier.pkl

# Batch processing
python main.py --batch path/to/image/directory --output results.json
```

### Python API

```python
from main import LeafSpeciesIdentifier

# Initialize identifier
identifier = LeafSpeciesIdentifier()

# Process single image
result = identifier.process_image('path/to/image.jpg')
print(f"Species: {result['species']}")
print(f"Confidence: {result['species_confidence']}")

# Process batch
results = identifier.process_batch(['image1.jpg', 'image2.jpg'])
```

## Project Structure

```
leaf_species_identifier/
├── streamlit_app.py        # Streamlit web application
├── main.py                 # Main CLI application script
├── preprocessing.py        # Image preprocessing utilities
├── leaf_validator.py       # Leaf validation module
├── feature_extraction.py   # Feature extraction module
├── classification.py       # Classification module
├── requirements.txt        # Python dependencies
├── models/                 # Trained models directory
├── uploads/                # Temporary upload directory
└── README.md              # This file
```

## Web Application Features

- **Drag & Drop Upload**: Easy file upload with drag-and-drop support
- **Real-time Processing**: Instant results with progress indicators
- **Confidence Visualization**: Visual confidence bars and top predictions
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Comprehensive error messages and validation

## Dependencies

- Python 3.7+
- Streamlit
- OpenCV
- scikit-learn
- NumPy
- SciPy
- scikit-image
- Matplotlib
- joblib
- Pillow

## Model Training

To train custom models:

```python
from classification import PlantClassifier
from feature_extraction import FeatureExtractor

# Load your training data
# X_train: feature matrix
# y_train: species labels

# Initialize classifier
classifier = PlantClassifier(model_type='random_forest')

# Train model
results = classifier.train_traditional_ml(X_train, y_train)

# Save model
classifier.save_model('models/custom_classifier.pkl')
```

## Streamlit Features

- **Simple Upload**: Easy file upload interface
- **Real-time Processing**: Instant results with progress indicators
- **Confidence Visualization**: Visual confidence bars and top predictions
- **Interactive Charts**: Bar charts for top predictions
- **Technical Details**: Expandable section with full results
- **Error Handling**: Comprehensive error messages and validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
