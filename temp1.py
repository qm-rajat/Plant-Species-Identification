import cv2
import numpy as np
import sys
import os
from transformers import pipeline
sys.path.append('LOMAVE_workspace/LeafOrNot_workspace')
from leafdet.quick_rule import quick_leaf_rule

def calculate_gradient(blurred_image):
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute gradient direction (angle in degrees)
    gradient_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.uint8)
    
    # Normalize angles to 0, 45, 90, 135 degrees
    angle = gradient_direction % 180
    angle[angle < 0] += 180
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]
            
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0
    
    return suppressed

def thresholding(suppressed, low_threshold=50, high_threshold=150):
    strong_edges = (suppressed > high_threshold).astype(np.uint8) * 255
    weak_edges = ((suppressed >= low_threshold) & (suppressed <= high_threshold)).astype(np.uint8) * 255
    return strong_edges, weak_edges

def edge_tracking(strong_edges, weak_edges):
    rows, cols = strong_edges.shape
    final_edges = strong_edges.copy()
    
    # Define 8-connected neighborhood
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Use a queue for BFS-like traversal
    from collections import deque
    queue = deque()
    
    # Add all strong edges to the queue
    for i in range(rows):
        for j in range(cols):
            if strong_edges[i, j] == 255:
                queue.append((i, j))
    
    # Process the queue
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and weak_edges[nx, ny] == 255 and final_edges[nx, ny] == 0:
                final_edges[nx, ny] = 255
                queue.append((nx, ny))
    
    return final_edges

def custom_edge_detection(image):
    # Step 1: Preprocessing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

    # Step 2: Gradient Calculation
    gradient_magnitude, gradient_direction = calculate_gradient(blurred_image)

    # Step 3: Non-Maximum Suppression
    thinned_edges = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Step 4: Thresholding
    strong_edges, weak_edges = thresholding(thinned_edges)

    # Step 5: Edge Tracking by Hysteresis
    final_edges = edge_tracking(strong_edges, weak_edges)

    return final_edges

# Example usage
image_path = 'leaf 2.jpeg'
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image.")
else:
    # Perform custom edge detection
    edges = custom_edge_detection(image)
    
    # Classify if it's a leaf or not
    label, score, diag = quick_leaf_rule(image)
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Leaf Classification: {label} (Score: {score:.2f})")
    
    # If it's a leaf, identify the species using AI
    if label == 'leaf':
        try:
            # Load the image classification pipeline for plant species
            # Using a model trained on plant species identification
            classifier = pipeline("image-classification", model="nateraw/vit-base-beans")
            # This model is trained on bean plant diseases, but demonstrates the concept
            # For better results, you could use models like:
            # - "microsoft/DialoGPT-medium" (general)
            # - "google/vit-base-patch16-224" (general vision)
            # - Plant-specific models from Hugging Face
            results = classifier(image_path)
            top_result = results[0]
            print(f"AI Species Prediction: {top_result['label']} (Confidence: {top_result['score']:.2f})")
        except Exception as e:
            print(f"AI species identification failed: {e}")
    else:
        print("Not a leaf, skipping species identification.")
    
    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Detected Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

