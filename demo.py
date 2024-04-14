from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import global_hist_eq as ghe

L = ghe.L
figsize = (9.6,3.3)

# set the filepath to the image file
filename = "input_img.png"
# read the image into a PIL entity
img = Image.open(fp=filename)
# keep only the Luminance component of the image
bw_img = img.convert("L")
# obtain the underlying np array
img_array = np.array(bw_img)

#original image
hist,cdf = ghe.get_histogram_cdf_of_img(img_array)
plot.figure(0,figsize=figsize)
plot.suptitle("Original image with histogram and cdf")
plot.subplot(121)
plot.axis("off")
plot.imshow(img_array,cmap="gray")
plot.subplot(122)
plot.bar(range(L),hist,width=1,label="hist")
plot.plot(cdf*np.max(hist),color="r",label=f"cdf*{np.max(hist)}")
plot.legend()

#global histogram equalization
ghe_img_array = ghe.perform_global_hist_equalization(img_array)
hist,cdf = ghe.get_histogram_cdf_of_img(ghe_img_array)
plot.figure(1,figsize=figsize)
plot.suptitle("Global Histogram Equalization")
plot.subplot(121)
plot.axis("off")
plot.imshow(ghe_img_array,cmap="gray")
plot.subplot(122)
plot.bar(range(L),hist,width=1,label="hist")
plot.plot(cdf*np.max(hist),color="r",label=f"cdf*{np.max(hist)}")
plot.legend()

plot.show()