import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import os
import matplotlib.pyplot as plt


CLUSTERS = 6

def get_colour_pallate(imagePath):

    # Fetch Image

    image = cv2.imread(imagePath)

    # Make a copy of image to use for the color palette generation

    #image_copy = image_resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=100)
    width = image.shape[1]
    height = image.shape[0]
    ratio = width/height

    height = 100
    width = int(100 * ratio)

    dim = (width,height)
    image_copy = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(image_copy)

    # Since the K-means algorithm we're about to do,

    # is very labour intensive, we will do it on a smaller image copy

    pixelImage = image_copy.reshape((image_copy.shape[0] * image_copy.shape[1], 3))



    # We use the sklearn K-Means algorithm to find the color histogram

    # from our small size image copy

    clt = KMeans(n_clusters=CLUSTERS)

    clt.fit(pixelImage)

    # build a histogram of clusters and then create a figure

    # representing the number of pixels labeled to each color

    colours = clt.cluster_centers_.astype(int)
    colours = colours.astype(int)

    return clt, colours



def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar


imagepath = "./test.jpg"

clt, list = get_colour_pallate(imagepath)

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)


print(list)

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


