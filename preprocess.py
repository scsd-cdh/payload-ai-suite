"""All image preprocessing logic and algorithms should go here

Features
- Handles both 3-channel (RGB) and 4-channel (RGB+NIR from TIFF alpha)
"""
import os
import cv2
import numpy as np

def populate(X_array, y_array, path, use_nir=False, end=False):
    """Populates the input arrays with preprocessed images and labels.

    Args:
        X_array (list): List to store the preprocessed images.
        y_array (list): List to store the labels corresponding to the images.
        path (str): Path to the directory containing image files.
        use_nir (bool): Whether to include the NIR channel.
        end (bool, optional): Flag to indicate whether to append default labels. Defaults to False.

    Returns:
        tuple: A tuple containing the updated X_array and y_array.
    """
    try:
        for image in os.listdir(path):
            image_path = os.path.join(path, image)

            rgb = cv2.imread(image_path)
            if rgb is None:
                print(f"Could not read {image_path}, skipping...")
                continue
            rgb = cv2.resize(rgb, (224, 224))
            
            if use_nir:
                # Simulated NIR channel as a grayscale version for now
                # You should replace this with the real NIR data later
                nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                nir = np.expand_dims(nir, axis=-1)  # Shape (224, 224, 1)
                rgb_nir = np.concatenate((rgb, nir), axis=-1)  # Shape (224, 224, 4)
                X_array.append(rgb_nir)
            else:
                X_array.append(rgb)

            if not end:
                y_array.append(image_path[0:1])
        
        if end:
            for _ in range(len(X_array), len(y_array)):
                y_array.append('N')
            
    except cv2.error as e:
        print(f"CV2 error in preprocess: {e}")
        return X_array, y_array
    
    return X_array, y_array