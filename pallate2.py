import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import os


CLUSTERS = 5
OFFSET = 2


def get_colour_pallate(imagePath):

    # Fetch Image

    image = cv2.imread(imagePath)

    # Make a copy of image to use for the color palette generation

    image_copy = image_resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=100)


    # Since the K-means algorithm we're about to do,

    # is very labour intensive, we will do it on a smaller image copy

    pixelImage = image_copy.reshape((image_copy.shape[0] * image_copy.shape[1], 3))



    # We use the sklearn K-Means algorithm to find the color histogram

    # from our small size image copy

    clt = KMeans(n_clusters=CLUSTERS+OFFSET)

    clt.fit(pixelImage)



    # build a histogram of clusters and then create a figure

    # representing the number of pixels labeled to each color

    colours = clt.cluster_centers_

    return colours


imagepath = "./test.jpg"

list = get_colour_pallate(imagepath)

print(list)
