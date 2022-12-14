import sys
import cv2
import imutils
import numpy as np
from keras.models import load_model
from Image_prep_rgb import find_controller, adjust_skew, remove_background
from Find_digits_rgb import find_reading_contours
from Read_digits import convert_cnt_to_numbers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = 'C:/Users/yjias/Desktop/on.jpg'

# Read image, resize it, and extract h and s channels
image = cv2.imread(path)
image = imutils.resize(image, height=500)
red_image = image[:, :, 2]
ret, thresh_img = cv2.threshold(red_image, int(np.mean(red_image) + 30), 255, cv2.THRESH_BINARY)

controller_cnt = find_controller(thresh_img)
adjusted_controller_image = adjust_skew(controller_cnt, image)
cleaned_controller_image = remove_background(adjusted_controller_image)
reading_cnts = find_reading_contours(cleaned_controller_image)
if len(reading_cnts) == 0:
    print('Did not find reading') # will set image capture waiting here later
else:
    model = load_model("test_model.h5")
    reading = convert_cnt_to_numbers(reading_cnts, cleaned_controller_image, model)
    print(reading)
