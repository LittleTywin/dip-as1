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

plot.figure(1)
plot.suptitle("Original Image")
plot.subplot(121)
plot.axis("Off")
plot.title("Image")
plot.imshow(img_array, cmap="gray")

img_array_hist = ahe.get_histogram_of_img(img_array)
plot.subplot(122)
plot.title("Histogram")
plot.bar(np.array(range(256)),img_array_hist,width=1)

plot.show()