import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in)

    if model.output_shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

n_sample = 1000
X, y = make_circles(n_sample,
                    noise=0.03,
                    random_state=42)
circles = pd.DataFrame({'X0':X[:,0], 'X1':X[:,1], 'y':y})
#plt.scatter(circles['X0'], circles['X1'], c=y, cmap=plt.cm.RdYlBu)
#plt.show()
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(1)
])
model_1.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
#model_1.fit(X, y, epochs=100, verbose=1)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
])
model_2.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
#model_2.fit(X, y, epochs=100, verbose=1)

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
])
model_3.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
#model_3.fit(X, y, epochs=100)

model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])
model_4.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.01),
                metrics=['accuracy'])
#model_4.fit(X, y, epochs=100)

X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_5.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
#history = model_5.fit(X_train, y_train, epochs=100, callbacks=[lr_scheduler])
#pd.DataFrame(history.history).plot(figsize=(10, 7), xlabel='epochs')

model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu')
])
model_6.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
model_6.fit(X_train, y_train, epochs=100)
model_6.summary()
#lrs = 1e-4 * (10**(np.arange(100)/20))
#plt.figure(figsize=(10, 7))
#plt.semilogx(lrs, history.history['loss'])
#plt.xlabel = 'learning rate'
#plt.ylabel = 'loss'
#plt.title("Learning rate vs loss")
#plt.show()
#loss, accuracy = model_4.evaluate(X_test, y_test)
#print(f"Model loss on the test set:{loss}")
#print(f"Model accuracty on the test set: {100*accuracy:.2f}%")

#pd.DataFrame(history.history).plot()
#plt.title("training curve")
#plt.show()




