import os
import cv2
import matplotlib.pyplot as plt

image_dir = 'outputs\images'
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(img_file)
        plt.axis('off')
        plt.show()
    else:
        print(f"Could not read {img_file}")