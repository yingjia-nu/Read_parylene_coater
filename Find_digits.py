
import cv2
from imutils import contours



def find_reading_contours(cnt):
    display_cnts = []
    cnts_sub, ret = cv2.findContours(cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # display contours should be reasonably big
    cnt_sub = sorted(cnts_sub, key=cv2.contourArea, reverse=True)
    for c in cnt_sub[:10]:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 15 and w <= 80) and (h >= 40 and h <= 100):
            display_cnts.append(c)
    # filter out the reading contours
    (controller_width, controller_height) = cnt.shape
    reading_cnts = []
    for c in display_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if y < controller_height/3: # reading on top
            reading_cnts.append(c)

    if len(reading_cnts) > 0:
        reading_cnts = contours.sort_contours(reading_cnts, method='left-to-right')[0]
    else:
        print('Did not find reading') # will replace with a log alarm later

    return reading_cnts # return it anyway, even it could be empty

