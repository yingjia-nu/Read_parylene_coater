import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_moons

x, y = make_moons(
    n_samples=1000, noise=0.03, random_state=42
)
x_1 = x[:, 0]
x_2 = x[:, 1]
x_train, x_test = x[:900], x[900:]
y_train, y_test = y[:900], y[900:]

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(x_train/2, y_train,
          epochs=100,
          validation_data=(x_test/2, y_test))