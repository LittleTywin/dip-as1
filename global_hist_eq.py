import numpy as np

def get_histogram_cdf_of_img(
        img_array: np.ndarray,
)-> tuple:
    """
    Calculates histogram and cumulative density function of input image.
    """
    L=256
    hist = np.zeros(L,dtype=np.float32)
    cdf = np.zeros(L,dtype=np.float32)
    img_size = img_array.size
    for i in range(L):
        hist[i] = np.sum(img_array==i)/img_size
        cdf[i] = cdf[i-1]+hist[i]
    
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
    pass

def perform_global_hist_equalization(
        img_array: np.ndarray,
) -> np.ndarray:
    pass