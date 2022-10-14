import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/yjias/Desktop/parylene_display_28.jpg", 0)
#cv2.imshow("show", img)
#cv2.waitKey(0)

img = img.reshape(1, 28, 28, 1)
img = img/255.

from keras.models import load_model
model = load_model("test_model.h5")
prediction = model.predict(img)
print(prediction.argmax())
