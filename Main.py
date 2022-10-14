import sys
import cv2
import imutils
import numpy as np
from keras.models import load_model
from Image_prep import find_controller, adjust_skew
from Find_digits import find_reading_contours
from Read_digits import convert_cnt_to_numbers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = 'C:/Users/yjias/Desktop/parylene_1p7.jpg'

# Read image, resize it, and extract h and s channels
image = cv2.imread(path)
image = imutils.resize(image, height=500)
hsv_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h = hsv_color[:, :, 0]
s = hsv_color[:, :, 1]
# check if the exposure time is too long
if np.mean(h) < 70 or np.mean(s) < 70:
    print('The exposure time is too long. '
          f'The average of h channel is {np.mean(h)}.'
          f'The average of s channel is {np.mean(s)}.')
    sys.exit()

# default use channel h and threshold 80 to do contour
ret, thresh = cv2.threshold(h, 80, 255, cv2.THRESH_BINARY)

controller_cnt = find_controller(thresh)
adjusted_controller_image = adjust_skew(controller_cnt, thresh)
reading_cnts = find_reading_contours(adjusted_controller_image)
model = load_model("test_model.h5")
read_digits = convert_cnt_to_numbers(reading_cnts, adjusted_controller_image, model)
reading = 0
for i in range(len(read_digits)):
    reading += read_digits[len(read_digits) - i - 1] * (10 ** i)
print(reading)
#image2 = image.copy()
#index = -1
#thickness = 4
#color = (255,0,255)
#cv2.drawContours(image2, controller_cnt, index, color, thickness)
#cv2.polylines(image2, [approx], True, (0, 255, 255), thickness)
#cv2.imshow("edge", adjusted_controller_cnt)
#cv2.waitKey(0)