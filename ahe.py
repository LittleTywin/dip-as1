import numpy as np
#from matplotlib import pyplot as plot

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
    n_pixels = img_array.size
    #calculate histogram
    hist = np.zeros(L,dtype=np.float32)
    v = np.zeros(L,dtype=np.float32)
    equalization_transform = np.zeros(L,dtype=np.uint32)
    for k in range(L):
        hist[k] = np.sum(img_array==k)/n_pixels
        v[k] = v[k-1] + hist[k]
        equalization_transform[k] = round((v[k]-v[0])/(1-v[0])*(L-1))

    return equalization_transform

def perform_global_hist_equalization(
        img_array: np.ndarray,
) -> np.ndarray:
    """
    Performs global histogram equalization on the input image and returns the
    result. Uses get_equalization_transform_of_img to get the equalization transform.

    Args:
    img_array (numpy.ndarray): A 2d uint8 array representing an 8-bit grayscale
        image.
    
    Returns:
    equalized_img (numpy.ndarray):A 2d uint8 array representing the output image.

    """
    equalized_img = np.zeros(img_array.shape,dtype=np.uint8)
    L=256
    eq_tr = get_equalization_transform_of_img(img_array)
    for i in range(L):
        equalized_img[img_array==i] = eq_tr[i]
    
    return equalized_img