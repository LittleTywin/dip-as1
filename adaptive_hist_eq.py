import numpy as np
from typing import Dict, Tuple
import math
import global_hist_eq as ghe

region_len_h = 64
region_len_w = 48

def get_region_pixel_indices_of_image(
        img_array: np.ndarray,
        region_len_h:int,
        region_len_w:int,
        region:Tuple[int,int]
) -> Tuple[range,range]:
    region_pixel_h_inds_start = region[0]*region_len_h
    region_pixel_h_inds_end = np.clip(
        (region[0]+1)*region_len_h,
        a_max=img_array.shape[0]
    )
    region_pixel_h_inds = range(
        region_pixel_h_inds_start,region_pixel_h_inds_end
    )
    region_pixel_w_inds_start = region[1]*region_len_w
    region_pixel_w_inds_end = np.clip(
        (region[1]+1)*region_len_w,
        a_max=img_array.shape[1]
    )
    region_pixel_w_inds = range(
        region_pixel_w_inds_end,region_pixel_w_inds_end
    )
    return region_pixel_h_inds,region_pixel_w_inds

def calculate_eq_transformations_of_regions(
        img_array: np.ndarray,
        region_len_h: int,
        region_len_w: int,
) -> Dict[Tuple, np.ndarray]:
    """
    Splits the img_array into contectual regions (rectangles 
    region_len_h x region_len_w) and then calculates and returns the histogram
    equalization transform in each of the regions.

    Contectual regions are represented with the vertice closer to the origin of
    the axes.

    Args:
    img_array(numpy.ndarray): A numpy array with ndim=2, dtype=numpy.uint8
        representing an 8-bit grayscale image
    region_len_h(int): The height of the contectual regions.
    region_len_w(int): The width of the contectual region.

    Returns:
    Dict[Tuple(int,int), numpy.ndarray]: A dictionary with region-hist_eq_tr
    pairs. 
    """
    region_to_eq_transform = {}

    img_h,img_w = img_array.shape
    region_h_ind_max = math.floor(img_h/region_len_h)
    region_w_ind_max = math.floor(img_w/region_len_w)
    for region_h_ind in range(region_h_ind_max):
        for region_w_ind in range(region_w_ind_max):
            region = region_h_ind,region_w_ind
            part_pixel_indices = get_region_pixel_indices_of_image(
                img_array,
                region_len_h,
                region_len_w,
                region,
            )
            img_array_part = img_array[part_pixel_indices]
            

            region_to_eq_transform[region] = ghe.get_equalization_transform_of_img(
                img_array_part)

    return region_to_eq_transform

def perform_adaptive_hist_equalization(
        img_array: np.ndarray,
        region_len_h:int,
        region_len_w:int,
) -> np.ndarray:
    """
    Accepts an input image and the size of the contectual region and performs
    adaptive histogram equalization with bilinear interpolation.

    Args:
    img_array(numpy.ndarray): A numpy array with ndim=2, dtype=numpy.uint8
        representing the input 8-bit grayscale image.
    region_len_h(int): The height of the contectual region.
    region_len_w(int): The width of the contectual region.

    Returns:
    equalized_img(numpy.ndarray): A numpy array with ndim=2, dtype=numpy.uint8
    representing the grayscale, 8-bit image produced after applying the global
    histogram equalization algorithm on the input image.
    """
    equalized_img = np.zeros(img_array.shape)
    return equalized_img