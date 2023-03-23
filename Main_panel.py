import sys
import cv2
import imutils
import numpy as np
from imutils import contours
from keras.models import load_model
from Image_prep_panel import find_controller, corner_check, adjust_skew, remove_background, controller_image_resize
from Find_digits_panel import find_reading_contours
from Read_digits import convert_cnt_to_numbers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# start a while True loop here. Take a picture, wait 1min, save the picture
path = 'C:/Users/yjias/Desktop/1.jpg'
image = cv2.imread(path)
# find the contour of the blue panel, offset -50 for white cleanroom

panel_cnt = find_controller(image, offset=-50)[0]

# check if 4 corners in case panel is blocked
if corner_check(panel_cnt):
    # adjust skew
    adjusted_panel_image = adjust_skew(panel_cnt, image, edge_size=20)
    (x, y, d) = adjusted_panel_image.shape
    if y < 450:
        print("camera is too far away") # add to alarm log, save image, and break out loop
    else:
        # find the controller contours
        controller_cnts = find_controller(adjusted_panel_image, offset=50)[:4]
        controller_cnts = contours.sort_contours(controller_cnts, method="left-to-right")[0]
    # use the right-most controller to check if the equipment is on or not

        last_cnt = controller_cnts[3]
        last_controller_image = controller_image_resize(last_cnt, adjusted_panel_image)
        cleaned_last_thresh = remove_background(last_controller_image)
        reading_last_cnts = find_reading_contours(cleaned_last_thresh, iter=2)
        if len(reading_last_cnts) == 0:
            print('Equipment is off') # add continue loop here
        else:
            model = load_model("test_model.h5")
            read_last = convert_cnt_to_numbers(reading_last_cnts, cleaned_last_thresh, model)
            print(f"vacuum reading is {read_last}") # replace by saving reading to a file
            for i in range(3):
                controller_cnt = controller_cnts[i]
                controller_image = controller_image_resize(controller_cnt, adjusted_panel_image)
                cleaned_controller_thresh = remove_background(controller_image)
                reading_cnts = find_reading_contours(cleaned_controller_thresh, iter=2)
                read_number = convert_cnt_to_numbers(reading_cnts, cleaned_controller_thresh, model)
                print(f"reading is {read_number}")
                if read_number == -100:
                    print(f"Adjust the capture angle for controller {i+1}") # replace with
                    # warning "adjust angle" and save image

else: # can't find panel, wait 5 min, counter +1, if counter<10, continue, if counter = 10,
    # add alarm log "blocked or too close", save image and break out loop
    print('Panel is blocked or camera is too close')

# when break out the loop, print("Adjust camera position")



