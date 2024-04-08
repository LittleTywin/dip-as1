import numpy as np
from matplotlib import pyplot as plot

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

    L = 256
    #calculate histogram
    hist = np.zeros(L,dtype=np.uint32)
    vk = np.zeros(L,dtype=np.uint32)
    equalization_transform = np.zeros(L,dtype=np.uint32)
    for k in range(L):
        hist[k] = np.sum(img_array==k)
        vk[k] = vk[k-1]+hist[k]
        equalization_transform[k] = round((vk[k]-vk[0])/(1-vk[0])*(1-L))

    return equalization_transform
