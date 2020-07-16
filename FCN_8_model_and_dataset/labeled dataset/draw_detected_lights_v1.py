#import argparse
import numpy as np
import cv2
#from scipy.misc import imresize
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
from keras.models import load_model
from PIL import Image


#argparser = argparse.ArgumentParser(
    #description='test FCN8 network for taillights detection')


#argparser.add_argument(
    #'-i',
    #'--image',
    #help='path to image file')




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
    model = load_model('full_CNN_model.h5')
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

    img_y_cr_cb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    #y_eq = cv2.equalizeHist(y)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_eq = clahe.apply(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    small_img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    
    
    #small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    #new_image = imresize(prediction, (rows, cols, 3))

    mask = cv2.resize(prediction, (cols, rows))
    
    mask =  np.expand_dims(mask, 2) 
    mask = np.repeat(mask, 3, axis=2) # give the mask the same shape as your image
    colors = {"red": [1.0,0.8,0.], "blue": [0.,0.,0.1]} # a dictionary for your colors, experiment with the values
    colored_mask = np.multiply(mask, colors["red"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
    image = image+colored_mask # element-wise sum (sinc img and mask have the same shape)

        
    # Add lane prediction to list for averaging
    #lanes.recent_fit.append(prediction)
    # Only using last five for average
    #if len(lanes.recent_fit) > 5:
        #lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    #lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    #blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    #lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    #lane_image = imresize(lane_drawn, (720, 1280, 3))

    # Merge the lane drawing onto the original image
    #result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    #return image.astype(float) / 255

    #return new_image
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

#cv2.imshow('histogram equalisation', detected)
#cv2.waitKey(0)

#if __name__ == '__main__':
    #args = argparser.parse_args()
    #_main_(args)
