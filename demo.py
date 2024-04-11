from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import ahe

# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp=filename)
# plot.figure
# plot.title("Input rgb image")
# plot.imshow(img)
# plot.show()

# keep only the Luminance component of the image
bw_img = img.convert("L")
# plot.figure
# plot.title("Input grayscale image")
# plot.imshow(bw_img,cmap="gray")
# plot.show()

# obtain the underlying np array
img_array = np.array(bw_img)

equalized_img_array = ahe.perform_global_hist_equalization(img_array)

hist = ahe.get_hist(img_array)
#plot.figure(1)
fig1, (ax1, ax2) = plot.subplots(ncols=2)
ax1.axis("off")
ax1.imshow(img_array, cmap="gray")
ax2.plot(hist)
#ax1 = plot.subplot(211)
#plot.axis("off")
#plot.imshow(img_array,cmap="gray")
#plot.subplot(212,sharex=ax1)
#plot.plot(hist)

# fig,ax = plot.subplots(2,2)
# ax[0,0].axis('off')
# ax[0,0].set_title("Original Image")
# ax[0,0].imshow(img_array, cmap="gray")
# hist = ahe.get_hist(img_array)
# ax[0,1].set_title("Original Image Histogram")
# ax[0,1].plot(hist)
# ax[1,0].axis('off')
# ax[1,0].set_title("Equalized Image")
# ax[1,0].imshow(equalized_img_array, cmap="gray")
# hist = ahe.get_hist(equalized_img_array)
# ax[1,1].set_title("Equalized Image Histogram")
# ax[1,1].plot(hist)
plot.show()
