import sys
import cv2
import numpy as np
import math

def order_points(pts):
    # initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will hav e the largest sum
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    pts_2 = np.delete(pts, [np.argmin(s), np.argmax(s)], axis=0)
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts_2, axis=1)
    tr = pts_2[np.argmin(diff)]
    bl = pts_2[np.argmax(diff)]
    # check if distance (tr-br)<(tr-tl). For rectangle shape of parylene panel, height < width
    width = math.dist(tr, tl)
    height = math.dist(tr, br)
    # if width > height, the above ordering assumption holds
    if width > height:
        rect[0] = tl
        rect[1] = tr
        rect[2] = br
        rect[3] = bl
    # otherwise, need to re-order the points. First determine which direction to rotate
    else:
        # look at the slope of the width edge. slope + or - depends on Edge_x * edge_y + or -
        edge = tr - br
        # re-order the points
        if edge[0] * edge[1] > 0:
            rect[0] = tr
            rect[1] = br
            rect[2] = bl
            rect[3] = tl
        else:
            rect[0] = bl
            rect[1] = tl
            rect[2] = tr
            rect[3] = br

    return rect

def four_point_transform(pts, image):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = int(max(widthA, widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1]-br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_controller(image, offset):
    red_image = image[:, :, 2]
    ret, thresh = cv2.threshold(red_image, int(np.mean(red_image) + offset), 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 120, 220, 255)
    # dilate edge for edge curve continuous and smooth
    kernel = np.ones((3, 3), 'uint8')
    edge_improved = cv2.dilate(edges, kernel, iterations=1)
    #cv2.imshow('Edge', edge_improved)
    #cv2.waitKey(0)
    cnts, ret = cv2.findContours(edge_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    return cnts

def corner_check(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    if len(approx) != 4:
        return False
    else:
        return True

def adjust_skew(cnt, image, edge_size):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    corners = np.zeros((4, 2), dtype='float32')
    for i in range(4):
        corners[i] = approx[i][0]
    warped = four_point_transform(corners, image)

    corrected_img = warped[edge_size:-edge_size, edge_size:-edge_size, :]
    #cv2.imshow("Warped", corrected_img)
    #cv2.waitKey(0)
    return corrected_img

def controller_image_resize(cnt, image):
    x, y, w, h = cv2.boundingRect(cnt)
    controller = image[y:y + h, x:x + w]
    controller = cv2.resize(controller, (300, 300), interpolation=cv2.INTER_CUBIC)
    controller = controller[10:-10, 10:-10, :]
    return controller

def remove_background(img):
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]
    red_blue_diff = np.mean(red) - np.mean(blue)
    if red_blue_diff >= 0:
        th = 200
    else:
        th = 180

    ret, thresh_red = cv2.threshold(red, th, 255, cv2.THRESH_BINARY)
    ret, thresh_green = cv2.threshold(green, th, 255, cv2.THRESH_BINARY)
    ret, thresh_blue = cv2.threshold(blue, th, 255, cv2.THRESH_BINARY)

    thresh = thresh_red - thresh_green
    cv2.imshow('tresh', thresh)
    cv2.waitKey(0)
    return thresh

#if __name__ == "__main__":

