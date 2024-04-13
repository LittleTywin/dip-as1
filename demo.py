from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import ahe

# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp=filename)
img.show()

# keep only the Luminance component of the image
bw_img = img.convert("L")

# obtain the underlying np array
img_array = np.array(bw_img)

#calculate and present original image along its histogram and cdf
img_array_hist, img_array_cdf = ahe.get_histogram_of_img(img_array)

fig1,ax1 = plot.subplots(1,2,figsize = (9.5,3.3))
fig1.suptitle("Original Image")
ax1[0].axis("off")
ax1[0].imshow(img_array, cmap="gray")
ax1[1].bar(np.array(range(ahe.L)), img_array_hist, width=1, label="pdf")
ax1[1].plot(img_array_cdf*np.max(img_array_hist), 'red', label="cdf")
ax1[1].legend()
#fig1.savefig("save_test.png")

#calculate and present image after global histogram equalization along with
#it's histogram and cdf
equalized_img_array = ahe.perform_global_hist_equalization(img_array)
equalized_img_hist,equalized_img_cdf = ahe.get_histogram_of_img(equalized_img_array)

fig2,ax2 = plot.subplots(1,2,figsize = (9.5,3.3))
fig2.suptitle("Global Histogram Equalization")
ax2[0].axis("off")
ax2[0].imshow(equalized_img_array, cmap="gray")
ax2[1].bar(np.array(range(ahe.L)), equalized_img_hist, width=1, label="pdf")
ax2[1].plot(equalized_img_cdf*np.max(equalized_img_hist), 'red', label="cdf")
ax2[1].legend()

plot.show()