import numpy as np
import cv2
from imutils import contours



def find_reading_contours(thresh, iter=1):
    edged = cv2.Canny(thresh, 120, 200, 255)
    # dilate edge for continuity and smoothness
    kernel = np.ones((3, 3), 'uint8')
    edge_imp = cv2.dilate(edged, kernel, iterations=iter)
    #cv2.imshow("edge", edge_imp)
    #cv2.waitKey(0)
    # find contours
    cnts, ret = cv2.findContours(edge_imp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display_cnts = []
    # display contours should be reasonably big
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:10]:
        (x, y, w, h) = cv2.boundingRect(c)
        if (15 <= w <= 100) and (40 <= h <= 150) and (5 < y < 100) and (15 < x+w < 270):
            display_cnts.append(c)
    #print(len(display_cnts))
    # filter out the reading contours
    reading_cnts = []
    for c in display_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if y < 100: # reading on top
            reading_cnts.append(c)
    #print(f"There are {len(reading_cnts)} digits")
    if len(reading_cnts) > 0:
        reading_cnts = contours.sort_contours(reading_cnts,method="left-to-right")[0]

    return reading_cnts # return it anyway, even it could be empty

