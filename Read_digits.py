import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# convert the cnt to (28x28) image for model input
def input_prep(cnt, image):
    x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)
    # don't corp too tight
    dig = image[(y_d - 5):(y_d + h_d + 5), (x_d - 5):(x_d + w_d + 5)]
    pix = max(dig.shape[0], dig.shape[1])
    # create a square background and add the digit image
    backg = np.zeros((pix, pix))
    backg_to_add = backg[:dig.shape[0], :dig.shape[1]]
    added = cv2.add(dig, backg_to_add, dtype=cv2.CV_32F)
    backg[int(pix / 2 - dig.shape[0] / 2):int(pix / 2 + dig.shape[0] / 2),
    int(pix / 2 - dig.shape[1] / 2):int(pix / 2 + dig.shape[1] / 2)] = added

    dig_28 = cv2.resize(backg, (28, 28), interpolation=cv2.INTER_NEAREST)
    dig_28 = dig_28.reshape(1, 28, 28, 1)
    dig_28 = dig_28 / 255.

    return dig_28



def convert_cnt_to_numbers(cnts, image, model):
    read_digits = []
    if len(cnts) > 0:
        for i in range(len(cnts)):
            # convert the cnt to (28x28) image for model input
            dig_cnt = cnts[i]
            dig_28 = input_prep(dig_cnt, image)
            pred = model.predict(dig_28)
            if pred.max() < 0.70:
                print('not sure about this digit') # replace with warning and save image
            read_digits.append(pred.argmax())
            reading = 0
            for i in range(len(read_digits)):
                reading += read_digits[len(read_digits) - i - 1] * (10 ** i)
    else:
         reading = -100

    return reading

