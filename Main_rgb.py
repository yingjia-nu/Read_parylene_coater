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

path = 'C:/Users/yjias/Desktop/parylene_1p.jpg'

# Read image, resize it, and extract h and s channels
image = cv2.imread(path)
image = imutils.resize(image, height=500)
red_image = image[:, :, 2]
ret, thresh_img = cv2.threshold(red_image, int(np.mean(red_image) + 30), 255, cv2.THRESH_BINARY)

controller_cnt = find_controller(thresh_img)
adjusted_controller_image = adjust_skew(controller_cnt, image)
cleaned_controller_image = remove_background(adjusted_controller_image)
reading_cnts = find_reading_contours(cleaned_controller_image)
model = load_model("test_model.h5")
read_digits = convert_cnt_to_numbers(reading_cnts, cleaned_controller_image, model)
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