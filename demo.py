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
plot.imshow(img_array, cmap="gray")

img_array_hist, img_array_cdf = ahe.get_histogram_of_img(img_array)
plot.subplot(122)
plot.bar(np.array(range(ahe.L)),img_array_hist,width=1,label="pdf")
plot.plot(img_array_cdf*np.max(img_array_hist),'red',label="cdf")
plot.legend(loc='best')

plot.figure(2)
plot.suptitle("Global Histogram Equalization")

equalized_img = ahe.perform_global_hist_equalization(img_array)
plot.subplot(121)
plot.axis("off")
plot.imshow(equalized_img,cmap="gray")

equalized_img_hist,equalized_img_cdf = ahe.get_histogram_of_img(equalized_img)
plot.subplot(122)
plot.bar(np.array(range(ahe.L)), equalized_img_hist, width=1, label="pdf")
plot.plot(equalized_img_cdf*np.max(equalized_img_hist),'red',label="cdf")
plot.legend(loc='best')


plot.show()