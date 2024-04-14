import numpy as np

L=256

def get_histogram_cdf_of_img(
        img_array: np.ndarray,
)-> tuple:
    """
    Calculates histogram and cumulative density function of input image.
    """
    hist = np.zeros(L,dtype=np.int32)
    cdf = np.zeros(L,dtype=np.float32)
    img_size = img_array.size
    for i in range(L):
        hist[i] = np.sum(img_array==i)
        cdf[i] = cdf[i-1]+hist[i]/img_size
    
    return hist,cdf

def get_equalization_transform_of_img(
        img_array: np.ndarray,
) -> np.ndarray:
    """
    Args:
    img_array(numpy.ndarray): A numpy array with ndim=2, dtype=numpy.uint8
        representing an 8-bit grayscale image
    
    Returns:
    equalization_transform(numpy.ndarray): A numpy array with ndim=1,
    dtype=numpy.uint8
    """
    equalization_transform = np.zeros(L, dtype=np.uint8)
    _,cdf = get_histogram_cdf_of_img(img_array)
    for i in range(L):
        equalization_transform[i] = round((cdf[i]-cdf[0])/(1-cdf[0])*(L-1))

    return equalization_transform

def perform_global_hist_equalization(
        img_array: np.ndarray,
) -> np.ndarray:
    ret_img_array = np.zeros(img_array.shape,dtype=np.uint8)
    equalization_transform = get_equalization_transform_of_img(img_array)
    for i in range(L):
        ret_img_array[img_array==i] = equalization_transform[i]
    
    return ret_img_array