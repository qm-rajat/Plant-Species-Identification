#!/usr/bin/env python3
"""
Flask Web Application for Plant Species Identification
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from plant_species_id.main_pipeline import identify_plant_species
import logging
from pathlib import Path

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Please upload an image file.'
            }), 400

        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save file
        file.save(filepath)

        # Process image
        result = identify_plant_species(filepath, debug=False)

        # Add image URL to result
        result['image_url'] = f'/static/uploads/{unique_filename}'

        # Clean up old files (keep only last 50 uploads)
        cleanup_old_files()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}'
        }), 500

def cleanup_old_files():
    """Clean up old uploaded files to save disk space"""
    try:
        upload_dir = Path(UPLOAD_FOLDER)
        files = list(upload_dir.glob('*'))

        # Sort by modification time, keep newest 50
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if len(files) > 50:
            for old_file in files[50:]:
                try:
                    old_file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {old_file}: {e}")

    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'plant-species-identification'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
