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

ahe_img = ahe.perform_adaptive_hist_equalization_no_interpolation(img_array,int(img_array.shape[0]/5),int(img_array.shape[1]/5))
plot.figure(1)
plot.imshow(ahe_img,cmap="gray")
plot.figure(2)
plot.imshow(img_array,cmap="gray")
plot.show()