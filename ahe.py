import numpy as np

def get_equalization_transform_of_img(
        img_array: np.ndarray,
) -> np.ndarray:
    """
    Takes the input image and calculates its histogramn then calculates the
    equalization transform
    
    Args:
    img_array (numpy.ndarray): A 2d, uint8 array representing an 8-bit grayscale
    image.
    
    Returns:
    equalization_transform (numpy.ndarray): An 1d, uint8 array of L elements

    """

