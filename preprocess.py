"""All image preprocessing logic and algorithms should go here

Features
- Basic reshaping and clean up of input data

TODO: 
- Create RGB-NIR fusion image
"""
import os
import cv2

def populate(X_array, y_array, path, end=False):
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
