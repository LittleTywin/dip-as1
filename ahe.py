import numpy as np

def get_histogram_of_img(
        img_array:np.ndarray,
) -> np.ndarray:
    """
    Calculates and returns the histogram of a grayscale image.
    
    Args:
    img_array (numpy.ndarray): A 2d, numpy.uint8 array representing the input
        image.
    
    Returns:
    hist (numpy.ndarray): A 1d, numpy.float32 array representing the histogram of
    the input image.

    """

    L=256
    hist = np.zeros(L,dtype=np.uint32)
    if not isinstance(img_array,np.ndarray):
        raise TypeError(f"img_array type should be 'numpy.ndarray',currently: {type(img_array)} ")
    if not img_array.dtype == np.uint8:
        raise ValueError(f"img_array dtype should be 'numpy.uint8', currently: 'numpy.{img_array.dtype}'")
    hist = np.zeros(L)

    for i in range(L):
        hist[i] = np.sum(img_array==i)
    return hist

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
    equalization_transform (numpy.ndarray): A numpy 1darray, dtype=numpyuint8
    of size L

    """
    pass