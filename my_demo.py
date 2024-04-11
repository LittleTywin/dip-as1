from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import ahe

# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp=filename)

# keep only the Luminance component of the image
bw_img = img.convert("L")

# obtain the underlying np array
img_array = np.array(bw_img)

eq_trans_of_regions = ahe.calculate_eq_transformations_of_regions(
    img_array,
    int(img_array.shape[0]/10),
    int(img_array.shape[1]/10),
)
pass