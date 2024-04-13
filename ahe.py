import numpy as np

L = 256

def get_histogram_of_img(
        img_array:np.ndarray,
) -> tuple:
    """
    Calculates and returns the histogram(probability density function) and cdf
    (cumulative density fuction) of a grayscale image.
    
    Args:
    img_array (numpy.ndarray): A 2d, numpy.uint8 array representing the input
        image.
    
    Returns:
    hist (numpy.ndarray): A 1d, numpy.float32 array representing the histogram of
    the input image.
    cdf (numpy.ndarray): A 1d, numpy.float32 array representing the cumulative
    density function of the histogram.
    """

    if not isinstance(img_array,np.ndarray):
        raise TypeError(f"img_array type should be 'numpy.ndarray',currently: {type(img_array)} ")
    if not img_array.dtype == np.uint8:
        raise ValueError(f"img_array dtype should be 'numpy.uint8', currently: 'numpy.{img_array.dtype}'")

    hist = np.zeros(L,dtype=np.float32)
    cdf = np.zeros(L,dtype=np.float32)
    
    for i in range(L):
        hist[i] = np.sum(img_array==i)/img_array.size
        cdf[i] = cdf[i-1]+hist[i]
    return hist,cdf

def get_equalization_transform_of_img(
        img_array:np.ndarray,
) -> np.ndarray:
    """
    Function calculates the histogram of the input image, and then creates
    the equalization_transform vector as follows:
        L = 256
        k = 0,1,...,L-1
        xk : discreet random variable [0,1,2,...,255]
        vk : cummulative density function
        yk : equalization transform
        yk = round(vk-v0)/(1-v0)*(L-1)
    
    Args:
    img_array (numpy.ndarray): A numpy 2darray, dtype=numpy.uint8 representing
    an 8-bit grayscale image.

    Returns:
    eq_transform (numpy.ndarray): A numpy 1darray, dtype=numpyuint8
    of size L

    """

    cdf = np.zeros(L,dtype=np.float32) #cumulative density function
    eq_transform = np.zeros(L,dtype=np.int32)
    hist,cdf = get_histogram_of_img(img_array)
    for i in range(L):
        eq_transform[i] = (cdf[i]-cdf[0])/(1-cdf[0])*(L-1)

    return eq_transform

def perform_global_hist_equalization(
        img_array:np.ndarray,
) -> np.ndarray:
    """
    Calculates the equalization_transformation with
    'get_equalization_transform_of_img(...)', applies it on 'img_array' and
    returns the result.

    Args:
    img_array(numpy.ndarray): 2d uint8 numpy matrix representing a grayscale image

    Returns:
    equalized_img(numpy.ndarray): 2d uint8 numpy matrix representing the input image
    after a global histogram equalization is applied

    """
    eq_transform = get_equalization_transform_of_img(img_array)
    equalized_img = np.ndarray(img_array.shape, dtype=np.uint8)
    for i in range(L):
        equalized_img[img_array==i] = eq_transform[i]
    
    return equalized_img
