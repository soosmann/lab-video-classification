"""
Process an image that we can pass to our networks.
"""
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def process_image(image, target_shape):
    """
    Given an image path, process it and return the array. 
    Includes resizing and normalization.
    
    Parameters:
        image (`str`): The path to the image to process
        target_shape (`tuple`): The target shape to resize to (height, width, channels)
    Returns:
        np.ndarray: The processed image as a numpy array
    """
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x
