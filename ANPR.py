
try:
    import cv2
except ImportError:
    print ("You must have OpenCV installed")
import matplotlib.pyplot as plt
import numpy as np

#Image(filename='../../../data/ANPR/sample_plates.png')

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



plt.rcParams['figure.figsize'] = 10, 10
plt.title('Sample Car')
image_path="out.jpg"
carsample=cv2.imread(image_path)
showfig(carsample,None)

plt.rcParams['figure.figsize'] = 7,7

# convert into grayscale
gray_carsample=cv2.cvtColor(carsample, cv2.COLOR_BGR2GRAY)
showfig(gray_carsample, plt.get_cmap('gray'))
# blur the image
blur=cv2.GaussianBlur(gray_carsample,(5,5),0)
showfig(blur, plt.get_cmap('gray'))
# find the sobel gradient. use the kernel size to be 3
sobelx=cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
showfig(sobelx, plt.get_cmap('gray'))
#Otsu thresholding
_,th2=cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
showfig(th2, plt.get_cmap('gray'))
#Morphological Closing
se=cv2.getStructuringElement(cv2.MORPH_RECT,(23,2))
closing=cv2.morphologyEx(th2, cv2.MORPH_CLOSE, se)
showfig(closing, plt.get_cmap('gray'))
_,contours,_=cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    cv2.drawContours(carsample, [box], 0, (0,255,0),2)
showfig(carsample, None)

def validate(cnt):
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    output=False
    width=rect[1][0]
    height=rect[1][1]
    if ((width!=0) & (height!=0)):
        if (((height/width>2) & (height>width)) | ((width/height>2) & (width>height))):
            if((height*width<16000) & (height*width>1000)):
                output=True

    return output

#Lets draw validated contours with red.
for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt)
        box=cv2.boxPoints(rect)
        box=np.int0(box)
        cv2.drawContours(carsample, [box], 0, (0,0,255),2)
showfig(carsample, None)
