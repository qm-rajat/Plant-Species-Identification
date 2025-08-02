"""
Main Flask Web Application for Plant Species Identification
Provides web interface for image upload and species prediction
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import time
from typing import Dict, Any, Optional

# Import our modules
from utils.preprocessing import ImagePreprocessor
from utils.edge_detection import EdgeDetector
from utils.feature_extraction import FeatureExtractor
from utils.leaf_validator import LeafValidator
from utils.classification import PlantClassifier, HybridClassifier
from utils.evaluation import ModelEvaluator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Global variables for models and processors
preprocessor = None
edge_detector = None
feature_extractor = None
leaf_validator = None
plant_classifier = None
hybrid_classifier = None
evaluator = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_models():
    """Initialize all models and processors"""
    global preprocessor, edge_detector, feature_extractor
    global leaf_validator, plant_classifier, hybrid_classifier, evaluator
    
    print("Initializing models and processors...")
    
    # Initialize processors
    preprocessor = ImagePreprocessor()
    edge_detector = EdgeDetector()
    feature_extractor = FeatureExtractor()
    evaluator = ModelEvaluator()
    
    # Initialize leaf validator
    leaf_validator = LeafValidator()
    
    # Try to load pre-trained models
    model_dir = 'models'
    if os.path.exists(model_dir):
        try:
            # Load leaf validator model if exists
            validator_path = os.path.join(model_dir, 'leaf_validator.pkl')
            if os.path.exists(validator_path):
                leaf_validator.load_model(validator_path)
                print("Loaded pre-trained leaf validator")
            
            # Load plant classifier if exists
            classifier_path = os.path.join(model_dir, 'plant_classifier.pkl')
            if os.path.exists(classifier_path):
                plant_classifier = PlantClassifier()
                plant_classifier.load_model(classifier_path)
                print("Loaded pre-trained plant classifier")
            else:
                # Create a dummy classifier for demo
                plant_classifier = PlantClassifier(model_type='random_forest')
                print("Created dummy plant classifier")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Create dummy models
            plant_classifier = PlantClassifier(model_type='random_forest')
    else:
        # Create dummy models
        plant_classifier = PlantClassifier(model_type='random_forest')
    
    # Initialize hybrid classifier
    hybrid_classifier = HybridClassifier(plant_classifier, leaf_validator)
    
    print("Models initialized successfully!")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string for web display"""
    if len(image.shape) == 3:
        # RGB image
        pil_image = Image.fromarray(image)
    else:
        # Grayscale image
        pil_image = Image.fromarray(image, mode='L')
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def process_uploaded_image(file_path: str) -> Dict[str, Any]:
    """Process uploaded image through the complete pipeline"""
    try:
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            return {'error': 'Could not load image'}
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocessing
        preprocess_results = preprocessor.preprocess_pipeline(
            file_path, add_noise=False, denoise=True
        )
        
        if preprocess_results is None:
            return {'error': 'Preprocessing failed'}
        
        processed_image = preprocess_results['final']
        
        # Edge detection
        edge_results = edge_detector.apply_all_methods(processed_image)
        
        # Feature extraction
        features = feature_extractor.extract_all_features(image)
        feature_vector = np.array(list(features.values()))
        
        # Leaf validation
        leaf_validation = leaf_validator.validate_with_rules(image)
        
        # Species classification (if it's a valid leaf)
        if leaf_validation['is_leaf']:
            try:
                if plant_classifier.is_trained:
                    classification_result = plant_classifier.predict_single(feature_vector)
                else:
                    # Dummy prediction for demo
                    classification_result = {
                        'predicted_class': 'Demo Species',
                        'confidence': 0.85,
                        'confidence_percent': 85.0,
                        'top_predictions': [
                            {'class': 'Demo Species', 'probability': 0.85, 'confidence': 85.0},
                            {'class': 'Alternative Species', 'probability': 0.10, 'confidence': 10.0},
                            {'class': 'Third Species', 'probability': 0.05, 'confidence': 5.0}
                        ]
                    }
            except Exception as e:
                print(f"Classification error: {e}")
                classification_result = {
                    'predicted_class': 'Classification Error',
                    'confidence': 0.0,
                    'confidence_percent': 0.0,
                    'top_predictions': []
                }
        else:
            classification_result = {
                'predicted_class': 'Not a leaf',
                'confidence': leaf_validation['confidence'],
                'confidence_percent': leaf_validation['confidence'] * 100,
                'top_predictions': []
            }
        
        # Prepare results
        results = {
            'success': True,
            'original_image': encode_image_to_base64(image),
            'processed_image': encode_image_to_base64(processed_image),
            'preprocessing': {
                'stages': list(preprocess_results.keys()),
                'final_size': processed_image.shape
            },
            'edge_detection': {
                'methods': list(edge_results.keys()),
                'images': {method: encode_image_to_base64(edges) 
                          for method, edges in edge_results.items() 
                          if not ('_x' in method or '_y' in method)}
            },
            'features': {
                'total_features': len(features),
                'categories': list(set([k.split('_')[0] for k in features.keys()])),
                'sample_features': dict(list(features.items())[:10])
            },
            'leaf_validation': leaf_validation,
            'classification': classification_result,
            'processing_time': time.time()
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Processing failed: {str(e)}'}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process image
        results = process_uploaded_image(file_path)
        
        if 'error' in results:
            flash(f"Error processing image: {results['error']}")
            return redirect(url_for('index'))
        
        # Store results in session or pass to template
        return render_template('results.html', results=results, filename=filename)
    
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save temporary file
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
    file.save(temp_path)
    
    try:
        # Process image
        results = process_uploaded_image(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        if 'error' in results:
            return jsonify(results), 500
        
        # Return simplified results for API
        api_results = {
            'leaf_validation': results['leaf_validation'],
            'classification': results['classification'],
            'features_extracted': results['features']['total_features'],
            'processing_successful': True
        }
        
        return jsonify(api_results)
        
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Demo page with sample images"""
    # Create some demo data
    demo_data = {
        'edge_methods': ['Sobel', 'Prewitt', 'Canny', 'LoG'],
        'feature_categories': ['Geometric', 'Texture', 'Color', 'Vein', 'Margin'],
        'sample_species': ['Oak', 'Maple', 'Birch', 'Pine', 'Willow']
    }
    return render_template('demo.html', demo_data=demo_data)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'preprocessor': preprocessor is not None,
            'edge_detector': edge_detector is not None,
            'feature_extractor': feature_extractor is not None,
            'leaf_validator': leaf_validator is not None,
            'plant_classifier': plant_classifier is not None and plant_classifier.is_trained,
            'hybrid_classifier': hybrid_classifier is not None
        },
        'timestamp': time.time()
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize models
    init_models()
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)