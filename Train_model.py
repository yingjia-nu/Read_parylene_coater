import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist

# import training and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# explore the dataset
image_index = 30
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()

# resize the dataset to meet keras requirements
img_rows = x_test.shape[1]
img_cols = x_test.shape[2]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train/255
x_test = x_test/255

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# build a model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation = 'relu',
                 input_shape = (img_rows, img_cols, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the model
batch_size = 128
epochs = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test,y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
model.save("test_model.h5")