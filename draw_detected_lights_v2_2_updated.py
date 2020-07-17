#import argparse
import numpy as np
import cv2
#from scipy.misc import imresize
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
from keras.models import load_model
#from PIL import Image
from scipy.ndimage.filters import median_filter

#argparser = argparse.ArgumentParser(
    #description='test FCN8 network for taillights detection')


#argparser.add_argument(
    #'-i',
    #'--image',
    #help='path to image file')

def auto_canny(image, sigma=0.33):
    

    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


# Load Keras model
#model = load_model('full_CNN_model.h5')

# Class to average lanes with
#class Lanes():
    #def __init__(self):
        #self.recent_fit = []
        #self.avg_fit = []

def taillight_detect(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    model = load_model('full_CNN_model_60_36.h5')
    #image1=image
    #image1=np.array(image1)
    #objects=np.squeeze(image,2)
    #rows,cols=objects.shape
    
    rows, cols,_ = image.shape
    
    #cols, rows = image.size
    #cols=160
    #rows=80
    # Get image ready for feeding into model
    
    small_img = cv2.resize(image, (160, 80))
    

    #img_y_cr_cb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
    #y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    #y_eq = cv2.equalizeHist(y)

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #y_eq = clahe.apply(y)

    #img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    #small_img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    
    #small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    #new_image = imresize(prediction, (rows, cols, 3))

    mask = cv2.resize(prediction, (cols, rows))
    
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    #y_eq = cv2.equalizeHist(y)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_eq = clahe.apply(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    image_he = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
    
    gray = cv2.cvtColor(image_he, cv2.COLOR_BGR2GRAY)
    
    #red_channel = image_he[:,:,2]
    
    #1
    #blurred = median_filter(gray, 3)
    #3,3,0
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #auto = auto_canny(blurred)

    
    #sobelx = cv2.Sobel(blurred,cv2.CV_64F,1,0,ksize=5) 
      
    # Calculation of Sobely 
    #sobely = cv2.Sobel(blurred,cv2.CV_64F,0,1,ksize=5) 
      
    # Calculation of Laplacian 
    #laplacian = cv2.Laplacian(blurred,cv2.CV_64F)
    
    #laplacian = cv2.convertScaleAbs(laplacian)
    
    
    #0.7
    #sharp = gray - 0.0001*laplacian
    #sharp=sharp.astype(np.uint8)
    #cv2.imshow('histogram equalisation', blurred)
    #cv2.waitKey(0)
    
    #plt.imshow(laplacian,cmap = 'gray')
    
    
    #for i in range(rows):
        #x = []
        #for j in range(cols):
            #k = gray[i,j]
            #print(k)
            #x.append(laplacian[i,j])
        #print(x)
    
    #auto = cv2.Canny(sharp, 10, 200)
    #auto = cv2.Canny(sharp, 100, 250)
    #auto=auto_canny(blurred)
    high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    auto = cv2.Canny(blurred, lowThresh, high_thresh,99)
    
    #cv2.imshow('histogram equalisation', auto)
    #cv2.waitKey(0)

    mask=mask.astype(np.uint8)
    for i in range(rows):
        for j in range(cols):
            if auto[i,j] >0 and mask [i,j]>1:
                auto[i,j]=255
            else:
                auto[i,j]=0
                
    #cv2.imshow('histogram equalisation', auto)
    #cv2.waitKey(0)
    
    #h, w = edges.shape[:2]
    filled_from_bottom = np.zeros((rows, cols))
    for col in range(cols):
        for row in reversed(range(rows)):
            if auto[row][col] < 255: filled_from_bottom[row][col] = 255
            else: break
    
    filled_from_top = np.zeros((rows, cols))
    for col in range(cols):
        for row in range(rows):
            if auto[row][col] < 255: filled_from_top[row][col] = 255
            else: break
    
    filled_from_left = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            if auto[row][col] < 255: filled_from_left[row][col] = 255
            else: break
    
    filled_from_right = np.zeros((rows, cols))
    for row in range(rows):
        for col in reversed(range(cols)):
            if auto[row][col] < 255: filled_from_right[row][col] = 255
            else: break
    
    for i in range(rows):
        for j in range(cols):
            if filled_from_bottom[i,j] ==0 and filled_from_top[i,j]==0 and filled_from_right[i,j] ==0 and filled_from_left[i,j]==0:
                auto[i,j]=mask[i,j]
            else:
                auto[i,j]=0
    
    for i in range(rows):
        for j in range(cols):
            if auto[i,j]>1:
                auto[i,j]=255
            else:
                auto[i,j]=0
    
    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(auto, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('histogram equalisation', closing)
    #cv2.waitKey(0)
    
    blanks = np.zeros_like(closing).astype(np.uint8)
    lane_drawn = np.dstack((closing, blanks, blanks))
    image = cv2.addWeighted(image, 1, lane_drawn, 1, 0)
    
    #cv2.imshow('histogram equalisation', image)
    #cv2.waitKey(0)
    
    #closing =  np.expand_dims(closing, 2) 
    #closing = np.repeat(closing, 3, axis=2) # give the mask the same shape as your image
    #colors = {"red": [0.0,1.0,1.0], "blue": [0.,0.,0.1]} # a dictionary for your colors, experiment with the values
    #colored_mask = np.multiply(closing, colors["red"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
    #image = image+colored_mask # element-wise sum (sinc img and mask have the same shape)
    #cv2.imshow('histogram equalisation', image)
    #cv2.waitKey(0)
    

    #return image.astype(float) / 255

    #return new_image
    #return auto
    return image
    
#lanes = Lanes()

# Where to save the output video
#vid_output = 'proj_reg_vid.mp4'

# Location of the input video
#clip1 = VideoFileClip("project_video.mp4")

#vid_clip = clip1.fl_image(road_lines)
#vid_clip.write_videofile(vid_output, audio=False)

#def _main_(args):
    #image_path   = args.image


#im = cv2.imread("ft.png")
#detected=taillight_detect(im)


#cv2.imwrite('detected.jpg',detected)



#image = cv2.imread("ft1.png")

#x=taillight_detect(image)

#cv2.imshow('histogram equalisation', x)
#cv2.waitKey(0)

#img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    #y_eq = cv2.equalizeHist(y)

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#y_eq = clahe.apply(y)

#img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
#image = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold

#wide = cv2.Canny(blurred, 10, 200)
#tight = cv2.Canny(blurred, 225, 250)
#auto = auto_canny(blurred)

# show the images
#cv2.imshow("Original", image)
#cv2.imshow("Edges", np.hstack([wide, tight, auto]))
#cv2.waitKey(0)
#rows,cols = auto.shape
#for i in range(rows):
        #x = []
        #for j in range(cols):
            #k = gray[i,j]
            #print(k)
            #x.append(auto[i,j])
        #print(x)

#cv2.imshow('histogram equalisation', detected)
#cv2.waitKey(0)

#if __name__ == '__main__':
    #args = argparser.parse_args()
    #_main_(args)
