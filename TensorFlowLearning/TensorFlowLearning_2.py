import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c='b',label='Training data')
    plt.scatter(test_data, test_labels, c='g',label='Testing data')
    plt.scatter(test_data, predictions, c='r', label='Predictions')
    plt.legend()
    plt.show()

def mae(y_test, y_pred):
    return tf.metrics.mean_absolute_error(y_test, y_pred)

def mse(y_test, y_pred):
    return tf.metrics.mean_squared_error(y_test, y_pred)


X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

X_train = X[:40]
y_train = y[:40]
X_test = X[40:]
y_test = y[40:]

#plt.figure(figsize=(10,7))
#plt.scatter(X_train, y_train, c='b', label='Training data')
#plt.scatter(X_test, y_test, c='g',label='Testing data')
#plt.legend()
#plt.show()

tf.random.set_seed(42)
model_1 = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
model_1.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=['mae'])
model_1.fit(tf.expand_dims(X_train, axis=-1),y_train, epochs=100)
y_preds_1 = model_1.predict(X_test)

tf.random.set_seed(42)
model_2 = tf.keras.Sequential([tf.keras.layers.Dense(1),
                               tf.keras.layers.Dense(1)])
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)
y_preds_2 = model_2.predict(X_test)
plot_predictions(X_train, y_train, X_test,y_test, y_preds_2)
mae_2 = mae(y_test, y_preds_2.squeeze()).numpy()
mse_2 = mse(y_test, y_preds_2.squeeze()).numpy()
print(mae_2, mse_2)
model_2.evaluate(X_test, y_test)

model_2.save("best_model_HDF5.h5")
load_model = tf.keras.models.load_model('best_model_HDF5.h5')
load_model.summary()




