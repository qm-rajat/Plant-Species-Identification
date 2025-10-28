#!/usr/bin/env python3
"""
Streamlit Web App for Leaf Species Identification
Provides a simple web interface for the leaf species identifier
"""

import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import tempfile

# Import the leaf species identifier
from main import LeafSpeciesIdentifier

# Page configuration
st.set_page_config(
    page_title="Leaf Species Identifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the identifier
@st.cache_resource
def load_identifier():
    return LeafSpeciesIdentifier()

identifier = load_identifier()

def main():
    st.title("🌿 Leaf Species Identifier")
    st.markdown("Upload a leaf image and our AI-powered system will validate if it's a leaf and identify the plant species with confidence scores.")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This application uses advanced computer vision and machine learning to:

    1. **Validate** if an image contains a leaf
    2. **Identify** the plant species
    3. **Provide** confidence scores and top predictions

    Built with Streamlit and scikit-learn.
    """)

    # File uploader
    st.subheader("📤 Upload Your Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, GIF, BMP. Maximum file size: 16MB."
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')

        with col2:
            st.subheader("🔍 Analysis Results")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                # Process the image
                with st.spinner("Analyzing image..."):
                    result = identifier.process_image(temp_path)

                # Clean up temp file
                os.unlink(temp_path)

                # Display results
                display_results(result)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

def display_results(result):
    """Display the analysis results"""

    # Leaf validation results
    st.subheader("✅ Leaf Validation")

    if result.get('is_leaf', False):
        st.success("✓ This image contains a valid leaf!")
        st.metric("Leaf Confidence", f"{result.get('leaf_confidence', 0)*100:.1f}%")

        # Progress bar for leaf confidence
        st.progress(result.get('leaf_confidence', 0))

        # Species identification results
        st.subheader("🎯 Species Identification")

        if 'species' in result and result['species'] != 'unknown_species':
            st.success(f"Predicted Species: **{result['species']}**")
            st.metric("Species Confidence", f"{result.get('species_confidence', 0)*100:.1f}%")

            # Progress bar for species confidence
            st.progress(result.get('species_confidence', 0))

            # Top predictions
            if 'top_predictions' in result and result['top_predictions']:
                st.subheader("📊 Top Predictions")

                # Create a table for top predictions
                pred_data = []
                for pred in result['top_predictions'][:5]:  # Show top 5
                    pred_data.append({
                        'Species': pred['class'],
                        'Confidence': f"{pred['probability']*100:.1f}%"
                    })

                if pred_data:
                    st.table(pred_data)

                    # Bar chart for top predictions
                    species_names = [p['class'] for p in pred_data]
                    confidences = [p['probability'] for p in pred_data]

                    st.bar_chart(data={'Confidence': confidences}, width='stretch')

        else:
            st.warning("⚠️ Unable to identify the species with sufficient confidence.")
            st.info("This might be due to the image quality, angle, or the species not being in our training data.")

    else:
        st.error("❌ This image does not appear to contain a valid leaf.")
        st.info("Please upload a clear image of a plant leaf for better results.")

    # Technical details (collapsible)
    with st.expander("🔧 Technical Details"):
        st.json(result)

if __name__ == "__main__":
    main()
