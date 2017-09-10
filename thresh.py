#!/usr/bin/env python3
from asyncio.windows_events import _WindowsDefaultEventLoopPolicy


try:
    import cv2
except ImportError:
    print ("You must have OpenCV installed")
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy import ndimage
from PIL import Image
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  io

import matplotlib.pyplot as plt
from skimage import data
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure

def faster_bradley_threshold(image, threshold=80, window_r=5):
    percentage = threshold / 100.
    window_diam = 2*window_r + 1
    # convert image to numpy array of grayscale values
    img = np.array(image.convert('L')).astype(np.float) # float for mean precision 
    # matrix of local means with scipy
    means = ndimage.uniform_filter(img, window_diam)
    # result: 0 for entry less than percentage*mean, 255 otherwise 
    height, width = img.shape[:2]
    result = np.zeros((height,width), np.uint8)   # initially all 0
    result[img >= percentage * means] = 255       # numpy magic :)
    # convert back to PIL image
    return Image.fromarray(result)

def bradley_threshold(image, threshold=75, windowsize=5):
    ws = windowsize
    image2 = copy.copy(image).convert('L')
    w, h = image.size
    l = image.convert('L').load()
    l2 = image2.load()
    threshold /= 100.0
    for y in range(h):
        for x in range(w):
            #find neighboring pixels
            neighbors =[(x+x2,y+y2) for x2 in range(-ws,ws) for y2 in range(-ws, ws) if x+x2>0 and x+x2<w and y+y2>0 and y+y2<h]
            #mean of all neighboring pixels
            mean = sum([l[a,b] for a,b in neighbors])/len(neighbors)
            if l[x, y] < threshold*mean:
                l2[x,y] = 0
            else:
                l2[x,y] = 255
            print ('working...'+str(x))
    return image2
def otsu_threshold(im):

    pixel_counts = [np.sum(im == i) for i in range(256)]

    s_max = (0,-10)
    ss = []
    for threshold in range(256):

        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])

        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # calculate
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        ss.append(s)

        if s > s_max[1]:
            s_max = (threshold, s)

    return s_max[0]

def showfig(image, ucmap):
    #There is a difference in pixel ordering in OpenCV and Matplotlib.
    #OpenCV follows BGR order, while matplotlib follows RGB order.
    if len(image.shape)==3 :
        b,g,r = cv2.split(image)       # get b,g,r
        image = cv2.merge([r,g,b])     # switch it to rgb
    imgplot=plt.imshow(image, ucmap)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    plt.show()
'otsu'
def threshold(t, image):
  intensity_array = []
  for w in range(0,image.size[1]):
    for h in range(0,image.size[0]):
      intensity = image.getpixel((h,w))
      if (intensity <= t):
        x = 0
      else:
        x = 255
      intensity_array.append(x)
  image.putdata(intensity_array)
  image.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
if __name__ == '__main__':
    img = Image.open('11.jpg').convert("L")
    #img.show()
    histo = img.histogram()
    #print(histo)
    histo_string = ''

    for i in histo:
      histo_string += str(i) + "\n"
    binary =  threshold(128,img)


    #print(histo_string)


 #   binary_otsu = threshold(100,img)
 #   binary_otsu .show()
#The aforementioned thresholding criterion is only suitable for the container image, in which the code character regions are darker than the surrounding regions.
   # threshold = 80
   # gray = rgb2gray(img)

   # print (width,height)
   # windows_size =int(width/30)

    t0 = time.process_time()
    threshed1 = faster_bradley_threshold(img)
    print('w/ numpy & scipy :', round(time.process_time()-t0, 3), 's')
    threshed1.show()


