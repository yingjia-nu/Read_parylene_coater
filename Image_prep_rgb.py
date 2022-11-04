import sys
import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1) # x+y for each point
    rect[0] = pts[np.argmin(s)] # top-left corner has the smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom right corner has the largest sum
    pts_2 = np.delete(pts, [np.argmin(s), np.argmax(s)], axis=0)

    diff = np.diff(pts_2, axis=1) # y-x for each point
    rect[1] = pts_2[np.argmin(diff)] # top-right corner has smallest y and biggest x
    rect[3] = pts_2[np.argmax(diff)] # bottom left corner has largest y and smallest x
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

def find_controller(thresh):
    edges = cv2.Canny(thresh, 120, 220, 255)
    # dilate edge for edge curve continuous and smooth
    kernel = np.ones((3, 3), 'uint8')
    edge_improved = cv2.dilate(edges, kernel, iterations=1)
    #cv2.imshow('Edge', edge_improved)
    #cv2.waitKey(0)
    cnts, ret = cv2.findContours(edge_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[0]



def adjust_skew(cnt, image):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    if len(approx) != 4:
        print("skew adjust failed. Didn't find the four corners")
        sys.exit()
    corners = np.zeros((4, 2), dtype='float32')
    for i in range(4):
        corners[i] = approx[i][0]

    warped = four_point_transform(corners, image)
    #cv2.imshow("Warped", warped)
    #cv2.waitKey(0)
    corrected_img = warped[5:-5, 5:-5, :]
    corrected_img = cv2.resize(corrected_img, (300, 300), interpolation=cv2.INTER_NEAREST)
    return corrected_img

def remove_background(img):
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]
    red_blue_diff = np.mean(red) - np.mean(blue)
    if red_blue_diff >= 15:
        th = 200
    elif red_blue_diff <= 5:
        th = 150
    else:
        th = 175
    print(f"th = {th}")
    ret, thresh_red = cv2.threshold(red, th, 255, cv2.THRESH_BINARY)
    ret, thresh_green = cv2.threshold(green, th, 255, cv2.THRESH_BINARY)
    ret, thresh_blue = cv2.threshold(blue, th, 255, cv2.THRESH_BINARY)

    thresh = thresh_red - thresh_green
    #cv2.imshow('tresh', thresh)
    #cv2.waitKey(0)
    return thresh

#if __name__ == "__main__":

