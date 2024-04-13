import numpy as np
def calculate_eq_transformations_of_regions(
        img_array: np.ndarray,
        region_len_h: int,
        region_len_w: int,
) -> dict:
    """
    This function splits the input image in (region_len_h x region_len_w)
    rectangles (contectual regions). Then calculates and returns the
    equalization transform for each one.

    Contectual regions are represented as a
    tuple with the coordinates of the region vertice closest to the origin of
    the axes(0,0)

    Args:
    img_array(numpy.ndarray): 2d uint8 matrix representing input image
    region_len_h(int): height of contectual region
    region_len_w(int): width of contectual region

    Returns:
    region_to_eq_transform(dict[Tuple,numpy.ndarray]): Contectual region - 
    equalization transform pairs. 
    """
    
    region_to_eq_transform = {}
    im_shape = img_array.shape
    region_h_max = int(im_shape[0]/region_len_h)
    region_w_max = int(im_shape[1]/region_len_w)
    for region_ind_h in range(region_h_max):
        for region_ind_w in range(region_w_max):
            pass
    return region_to_eq_transform