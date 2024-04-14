from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import global_hist_eq as ghe
import adaptive_hist_equ as ahe

region_len_h = 36
region_len_w = 48


# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp=filename)

# keep only the Luminance component of the image
bw_img = img.convert("L")

# obtain the underlying np array
img_array_tmp = np.array(bw_img)

#clip image in order to have equally sized contectual regions
img_array = img_array_tmp[
    0:int(img_array_tmp.shape[0]/region_len_h)*region_len_h,
    0:int(img_array_tmp.shape[1]/region_len_w)*region_len_w,
]

#calculate and present original image along its histogram and cdf
def original_image_hist_cdf():
    img_array_hist, img_array_cdf = ghe.get_histogram_of_img(img_array)

    fig1,ax1 = plot.subplots(1,2,figsize = (9.5,3.3))
    fig1.suptitle("Original Image")
    ax1[0].axis("off")
    ax1[0].imshow(img_array, cmap="gray")
    ax1[1].bar(np.array(range(ghe.L)), img_array_hist, width=1, label="pdf")
    ax1[1].plot(img_array_cdf*np.max(img_array_hist), 'red', label="cdf")
    ax1[1].legend()
    #fig1.savefig("save_test.png")

#calculate and present image after global histogram equalization along with
#it's histogram and cdf
def global_hist_equalization():
    equalized_img_array = ghe.perform_global_hist_equalization(img_array)
    equalized_img_hist,equalized_img_cdf = ghe.get_histogram_of_img(equalized_img_array)

    fig2,ax2 = plot.subplots(1,2,figsize = (9.5,3.3))
    fig2.suptitle("Global Histogram Equalization")
    ax2[0].axis("off")
    ax2[0].imshow(equalized_img_array, cmap="gray")
    ax2[1].bar(np.array(range(ghe.L)), equalized_img_hist, width=1, label="pdf")
    ax2[1].plot(equalized_img_cdf*np.max(equalized_img_hist), 'red', label="cdf")
    ax2[1].legend()

##
def global_hist_equalization_no_interp():
    ghe_cr = ahe.perform_adaptive_hist_equalization_no_interpolation(
        img_array,
        region_len_h,
        region_len_w
    )
    ghe_cr_hist, ghe_cr_cdf = ghe.get_histogram_of_img(ghe_cr)

    fig3,ax3 = plot.subplots(1,2,figsize = (9.5,3.3))
    fig3.suptitle("Global Histogram Equalization per Contectual Region")
    ax3[0].axis("off")
    ax3[0].imshow(ghe_cr, cmap="gray")
    ax3[1].bar(np.array(range(ghe.L)), ghe_cr_hist, width=1, label="pdf")
    ax3[1].plot(ghe_cr_cdf*np.max(ghe_cr_hist), 'red', label="cdf")
    ax3[1].legend()
##
def adaptive_hist_equalization():
    ahe_img = ahe.perform_adaptive_hist_equalization(
        img_array,
        region_len_h,
        region_len_w,
    )
    ahe_img_hist, ahe_img_cdf = ghe.get_histogram_of_img(ahe_img)
    fig4,ax4 = plot.subplots(1,2,figsize = (9.5,3.3))
    fig4.suptitle("Adaptive Histogram Equalization")
    ax4[0].axis("off")
    ax4[0].imshow(ahe_img, cmap="gray")
    ax4[1].bar(np.array(range(ghe.L)), ahe_img_hist, width=1, label="pdf")
    ax4[1].plot(ahe_img_cdf*np.max(ahe_img_hist), 'red', label="cdf")
    ax4[1].legend()



if __name__=="__main__":
    global_hist_equalization_no_interp()
    adaptive_hist_equalization()
    plot.show()