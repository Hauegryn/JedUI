import cv2
import numpy as np
import os

# Define directories
input_dir = "gesture_dataset"
output_dir = "preprocessed_dataset"
os.makedirs(output_dir, exist_ok=True)

# Target input dimensions for our neural network
IMG_SIZE = 224

def preprocess_image(image_path):
    """
    Loads a raw image, resizes it, normalizes pixels, and applies augmentations.
    """
    # 1. Load the raw image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None

    # 2. Resizing: Scale the image to fixed neural network dimensions (224x224)
    # Using INTER_AREA interpolation is best for shrinking images cleanly
    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 3. Normalization: Convert integer pixel values [0, 255] to floats [0.0, 1.0]
    normalized_img = resized_img.astype(np.float32) / 255.0

    # 4. Data Augmentation Example (Simulating Real-World Variances):
    # Let's apply a random brightness adjustment to simulate dim/bright restaurant lighting
    augmented_img = apply_random_brightness(normalized_img)

    return augmented_img

def apply_random_brightness(image):
    """
    Randomly scales pixel brightness to simulate restaurant lighting changes.
    """
    # Generate a random brightness factor between 0.8 (darker) and 1.2 (brighter)
    factor = np.random.uniform(0.8, 1.2)
    
    # Apply factor and clip values to stay strictly within the [0.0, 1.0] range
    bright_img = np.clip(image * factor, 0.0, 1.0)
    return bright_img

# --- Main Batch Processing Loop ---
print("--- Starting Preprocessing Pipeline ---")

if not os.path.exists(input_dir) or not os.listdir(input_dir):
    print(f"Error: No images found in '{input_dir}'. Run your capture script first!")
else:
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            
            # Run the preprocessing pipeline
            processed_img = preprocess_image(input_path)
            
            if processed_img is not None:
                # Convert back to [0, 255] integer scale just for saving/visualizing as an image file
                save_img = (processed_img * 255.0).astype(np.uint8)
                
                output_path = os.path.join(output_dir, f"prep_{filename}")
                cv2.imwrite(output_path, save_img)
                processed_count += 1

    print(f"Successfully preprocessed and saved {processed_count} images to '{output_dir}/'.")
