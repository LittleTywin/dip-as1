from PIL import Image
from matplotlib import pyplot as plot
import numpy as np
import global_hist_eq as ghe
import adaptive_hist_eq as ahe
import os

c="N"
if not os.path.isfile("demo.py"):
    print("You are not running the script from the project directory!")
    print("Random stuff might happen. Continue anyway? N/y")
    input(c)
    if c!='Y' and c!="y":
        exit        

L = ghe.L
region_len_h = ahe.region_len_h
region_len_w = ahe.region_len_w
figsize = (9.6,3.3)

# set the filepath to the image file
filename = "input_img.png"
# read the image into a PIL entity
img = Image.open(fp=filename)
# keep only the Luminance component of the image
bw_img = img.convert("L")
# obtain the underlying np array
img_array = np.array(bw_img)

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#original image
hist,cdf = ghe.get_histogram_cdf_of_img(img_array)
fig0,ax0 = plot.subplots(1,2,figsize=figsize)
fig0.suptitle("Original Image")
ax0[0].axis("off")
ax0[0].imshow(img_array,cmap="gray")
ax0[1].bar(range(L),hist/1000,width=1,label="hist")
ax0[1].set(xticks=np.arange(0,257,32))
ax01twinx = ax0[1].twinx()
ax01twinx.plot(cdf,color="r",label="cdf")
lines,labels = ax0[1].get_legend_handles_labels()
lines2,labels2 = ax01twinx.get_legend_handles_labels()
ax0[1].legend(lines+lines2,labels+labels2,loc="right")
ax0[1].set_ylabel("Samples (x1000)")
ax01twinx.set_ylabel("Cumulative Density Function")
fig0.savefig("output/original.png")

# global histogram equalization
ghe_img_array = ghe.perform_global_hist_equalization(img_array)
hist,cdf = ghe.get_histogram_cdf_of_img(ghe_img_array)
fig1,ax1 = plot.subplots(1,2,figsize=figsize)
fig1.suptitle("Global Histogram Equalization")
ax1[0].axis("off")
ax1[0].imshow(ghe_img_array,cmap="gray")
ax1[1].bar(range(L),hist/1000,width=1,label="hist")
ax1[1].set(xticks=np.arange(0,257,32))
ax11twinx = ax1[1].twinx()
ax11twinx.plot(cdf,color="r",label="cdf")
lines,labels = ax1[1].get_legend_handles_labels()
lines2,labels2 = ax11twinx.get_legend_handles_labels()
ax1[1].legend(lines+lines2,labels+labels2,loc='right',bbox_to_anchor=(1,.6))
ax1[1].set_ylabel("Samples (x1000)")
ax11twinx.set_ylabel("Cumulative Density Function")
fig1.savefig("output/ghe.png")

#global histogram equalization per region
ghe_per_region_img_array = ahe.perform_adaptive_hist_equalization_no_interp(
    img_array,region_len_h,region_len_w)
hist,cdf = ghe.get_histogram_cdf_of_img(ghe_per_region_img_array)
fig2,ax2=plot.subplots(1,2,figsize=figsize)
fig2.suptitle("Global Histogram Equalization Per Region")
ax2[0].imshow(ghe_per_region_img_array,cmap="gray")
ax2[0].axis("off")
ax2[1].bar(range(L),hist/1000,width=1,label="hist")
ax2[1].set(xticks=np.arange(0,257,32))
ax21twinx=ax2[1].twinx()
ax21twinx.plot(cdf,label="cdf",color='r')
lines,labels = ax2[1].get_legend_handles_labels()
lines2,labels2 = ax21twinx.get_legend_handles_labels()
ax2[1].legend(lines+lines2,labels+labels2,loc="upper center",bbox_to_anchor=(0.2,1))
ax2[1].set_ylabel("Samples (x1000)")
ax21twinx.set_ylabel("Cummulative Density Function")
fig2.savefig("output/ghe_per_region.png")

#adaptive histogram equalization
ahe_img_array = ahe.perform_adaptive_hist_equalization(
    img_array,region_len_h,region_len_w)
hist,cdf = ghe.get_histogram_cdf_of_img(ahe_img_array)
fig3,ax3=plot.subplots(1,2,figsize=figsize)
fig3.suptitle("Adaptive Histogram Equalization")
ax3[0].imshow(ahe_img_array,cmap="gray")
ax3[0].axis("off")
ax3[1].bar(range(L),hist/1000,width=1,label="hist")
ax3[1].set(xticks=np.arange(0,257,32))
ax31twinx=ax3[1].twinx()
ax31twinx.plot(cdf,label="cdf",color='r')
lines,labels = ax3[1].get_legend_handles_labels()
lines2,labels2 = ax31twinx.get_legend_handles_labels()
ax3[1].legend(lines+lines2,labels+labels2,loc="upper center",bbox_to_anchor=(0.2,1))
ax3[1].set_ylabel("Samples (x1000)")
ax31twinx.set_ylabel("Cummulative Density Function")
fig3.savefig("output/ahe.png")

#plot.show()