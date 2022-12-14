import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def view_random_image(target_dir, target_class):
    target_folder = target_dir + '/' + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + '/' + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.show()
    print(f"Image shape: {img.shape} ")

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    plt.figure(figsize=(10, 14))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, color='red', label='training_loss')
    plt.plot(epochs, val_loss, color='blue', label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracy, c='green', label='training_accuracy')
    plt.plot(epochs, val_accuracy, c='orange', label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def show_augment(data, data_augmented):
    images, labels = data.next()
    augmented_images, augmented_labels = data_augmented.next()

    random_number = random.randint(0, 31)
    plt.imshow(images[random_number])
    plt.title(f"Original image")
    plt.axis(False)
    plt.figure()
    plt.imshow(augmented_images[random_number])
    plt.title(f"Augmented image")
    plt.axis(False)
    plt.show()

def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)
    img = img/255.
    return img

def pred_and_plot(model, filename, class_names):
    img = load_and_prep_image(filename)
    origin_img = mpimg.imread(filename)
    pred = model.predict(img)
    pred_class = class_names[round(pred[0][0])]
    plt.imshow(origin_img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()

folder = 'C:/Users/yjias/Desktop/pizza_steak'
########### Explot the data
#for dirpath, dirnames, filenames in os.walk(folder):
#    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
data_dir = pathlib.Path(folder + '/train')
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
#print(class_names)
#view_random_image('C:/Users/yjias/Desktop/pizza_steak/train', 'steak')
############ training
tf.random.set_seed(42)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
train_datagen_augmented = ImageDataGenerator(rescale=1./255,
                                             rotation_range=20,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

train_dir = folder + '/train'
test_dic = folder + '/test'

print('Non-augmented training images:')
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               seed=42)
print('Augmented training images:')
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   batch_size=32,
                                                                   target_size=(224, 224),
                                                                   class_mode='binary',
                                                                   shuffle=True)
print('Unchanged test images:')
valid_data = valid_datagen.flow_from_directory(test_dic,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               seed=42)


#show_augment(train_data, train_data_augmented)
"""
model_5 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_5.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
history_5 = model_5.fit(train_data_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
plot_loss_curves(history_5)
model_5.save('pizza_steak_model.h5')
"""
model = tf.keras.models.load_model('pizza_steak_model.h5')
pred_and_plot(model, "C:/Users/yjias/Desktop/test_steak.png", class_names)

"""
model_4 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_4.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
history_4 = model_4.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
plot_loss_curves(history_4)
"""
"""
model_3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_3.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
history_3 = model_3.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

plot_loss_curves(history_3)
"""
"""
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_2.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
history_2 = model_2.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
print(model_2.summary())
"""
"""
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=(3, 3),
                           activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=(3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding='valid'),
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation='relu'),
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_1.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
history = model_1.fit(train_data,
                      epochs=5,
                      steps_per_epoch=len(train_data),
                      validation_data=valid_data,
                      validation_steps=len(valid_data))
"""
