#!/usr/bin/env python3
"""
Flask Web App for Leaf Species Identification
Provides a web interface for the leaf species identifier
"""

import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Import the leaf species identifier
from main import LeafSpeciesIdentifier

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'leaf-species-identifier-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the identifier
identifier = LeafSpeciesIdentifier()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Check if file was selected
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Check if file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                # Process the image
                result = identifier.process_image(temp_path)

                # Clean up temp file
                os.unlink(temp_path)

                # Store result in session for GET requests
                import json
                session['result'] = json.dumps(result, default=str)

                # Redirect to results page
                return redirect(url_for('show_result'))

            except Exception as e:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                flash(f'Error processing image: {str(e)}')
                return redirect(url_for('index'))

        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(url_for('index'))
    else:
        # GET request - redirect to home
        return redirect(url_for('index'))

@app.route('/result')
def show_result():
    """Show processing results"""
    import json
    result_json = session.get('result')
    if result_json:
        result = json.loads(result_json)
        return render_template('result.html', result=result)
    else:
        flash('No result available. Please upload an image first.')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """500 error handler"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
