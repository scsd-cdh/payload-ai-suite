"""All image preprocessing logic and algorithms should go here

Features
- Basic reshaping and clean up of input data

TODO:
- Create RGB-NIR fusion image
"""
import os
import cv2

def populate(X_array, y_array, path, end=False):
    """Populates the input arrays with preprocessed images and labels.

    Args:
        X_array (list): List to store the preprocessed images.
        y_array (list): List to store the labels corresponding to the images.
        path (str): Path to the directory containing image files.
        end (bool, optional): Flag to indicate whether to append default labels. Defaults to False.

    Returns:
        tuple: A tuple containing the updated X_array and y_array.
    """
    try:
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            modified_image = cv2.imread(image_path)
            modified_image = cv2.resize(modified_image, (224, 224) )
            X_array.append(modified_image)
            if not end:
                y_array.append((image_path[0:1]))

        if end:
            for i in range(1,99):
                y_array.append('N')
    except cv2.error:
        print("CV2 error in preprocess")
        return X_array, y_array
    return X_array, y_array
