import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
import sklearn.preprocessing

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.boston_housing.load_data(
        path='boston_housing.npz', test_split=0.2, seed=113)

x_train_normal = sklearn.preprocessing.normalize(x_train, norm='l2')
x_test_normal = sklearn.preprocessing.normalize(x_test, norm='l2')

tf.random.set_seed(42)
house_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
house_model.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    metrics=['mae'])
history = house_model.fit(x_train_normal, y_train, epochs=60, verbose=0)

pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
house_model.evaluate(x_test_normal, y_test)