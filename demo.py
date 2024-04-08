from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import ahe

# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp=filename)
plot.figure
plot.title("Input rgb image")
plot.imshow(img)
plot.show()

# keep only the Luminance component of the image
bw_img = img.convert("L")
plot.figure
plot.title("Input grayscale image")
plot.imshow(bw_img,cmap="gray")
plot.show()

# obtain the underlying np array
img_array = np.array(bw_img)

equalized_img_array = ahe.perform_global_hist_equalization(img_array)
plot.figure
plot.title("Equalized image")
plot.imshow(equalized_img_array,cmap="gray")
plot.show()
