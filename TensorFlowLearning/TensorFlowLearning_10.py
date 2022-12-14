import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pathlib
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def view_random_image(path, class_name):
    folder = path + class_name
    random_image = random.sample(os.listdir(folder), 1)
    img = mpimg.imread(folder + '/' + random_image[0])
    plt.imshow(img)
    plt.title(class_name)
    plt.show()
    print(img.shape)

def load_and_prep_image(filename, img_shape=112):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)
    img = img/255.
    return img

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    plt.figure(figsize=(8, 14))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='validation_loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='validation_accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Accuracy')
    plt.show()

def plot_epoch_lr(history):
    #pd.DataFrame(history.history).plot(figsize=(10, 7), xlabel='epochs')
    lrs = 1e-4 * (10**(np.arange(50)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history['loss'])
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title("Learning_rate vs. Loss")
    plt.show()

def pred_and_plot(model, filename, class_names):
    img = load_and_prep_image(filename)
    origin_img = mpimg.imread(filename)
    pred = model.predict(img)
    pred_class = class_names[round(pred[0][0])]
    plt.imshow(origin_img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()

train_dir = pathlib.Path('C:/Users/yjias/Desktop/Pictures/Train/')
test_dir = pathlib.Path('C:/Users/yjias/Desktop/Pictures/Test/')
class_names = [item.name for item in train_dir.glob("*")]
""" view images
for dirpath, dirnames, filenames in os.walk("C:/Users/yjias/Desktop/Pictures/"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")

view_random_image('C:/Users/yjias/Desktop/Pictures/Train/', 'ring')
view_random_image('C:/Users/yjias/Desktop/Pictures/Train/', 'rainbow')
"""
# pre-process images
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)
train_datagen_augmented = ImageDataGenerator(rescale=1./255,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(112, 112),
                                               class_mode='binary',
                                               batch_size=2)
test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(112, 112),
                                             class_mode='binary',
                                             batch_size=2)
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(112, 112),
                                                                   batch_size=2,
                                                                   class_mode='binary',
                                                                   shuffle=True)

# build model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='relu',
                           input_shape=(112, 112, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=0.002),
                metrics=['accuracy'])
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
history_1 = model_1.fit(train_data_augmented,
                        epochs=15,
                        #callbacks=[lr_scheduler],
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

plot_loss_curves(history_1)
# find the best learning rate
#plot_epoch_lr(history_1)
model_1.save('rainbow_ring.h5')
"""
# prediction
model_2 = tf.keras.models.load_model('rainbow_ring.h5')
pred_and_plot(model_2, 'C:/Users/yjias/Desktop/Pictures/Test/ring/15.jpg', class_names)
"""