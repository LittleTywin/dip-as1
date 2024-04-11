import numpy as np
#from matplotlib import pyplot as plot

def get_hist(img_array):
    L=256
    n_pixels = img_array.size
    hist = np.zeros(L,dtype=np.float32)
    for k in range(L):
        hist[k] = np.sum(img_array==k)/n_pixels
    return hist

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
    hist = np.zeros(L,dtype=np.float64)
    v = np.zeros(L,dtype=np.float64)
    equalization_transform = np.zeros(L,dtype=np.uint8)
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

def calculate_eq_transformations_of_regions(
        img_array:np.ndarray,
        region_len_h:int,
        region_len_w:int
) -> dict:
    eq_transformation_of_regions = {}
    for i in range(int(img_array.shape[0]/region_len_h)):
        for j in range(int(img_array.shape[1]/region_len_w)):
            region_matrix = img_array[i*region_len_h:(i+1)*region_len_h,j*region_len_w:(j+1)*region_len_w]
            eq_transformation_of_regions[(i,j)] = get_equalization_transform_of_img(region_matrix)
    return eq_transformation_of_regions

def perform_adaptive_hist_equalization(
        img_array:np.ndarray,
        region_len_h:int,
        region_len_w:int,
) -> np.ndarray:
    pass

def perform_adaptive_hist_equalization_no_interpolation(
        img_array:np.ndarray,
        region_len_h:int,
        region_len_w:int,
) -> np.ndarray:
    eq_tr_of_regions = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
    equalized_img_array = np.zeros(img_array.shape)
    for i in range(int(img_array.shape[0]/region_len_h)):
        for j in range(int(img_array.shape[1]/region_len_w)):
            equalized_img_part = np.zeros((region_len_h,region_len_w))
            equalized_img_array_part = np.zeros((region_len_h,region_len_w))
            for v in range(256):
                img_array_part = img_array[i*region_len_h:(i+1)*region_len_h,j*region_len_w:(j+1)*region_len_w]
                equalized_img_array_part[img_array_part==v] = eq_tr_of_regions[(i,j)][v]
            equalized_img_array[i*region_len_h:(i+1)*region_len_h,j*region_len_w:(j+1)*region_len_w] = equalized_img_array_part
    return equalized_img_array
            
