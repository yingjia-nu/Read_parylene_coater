import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pathlib
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
for dirpath, dirnames, filenames in os.walk("C:/Users/yjias/Desktop/10_food_classes_all_data"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
"""
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
    pred_class = class_names[pred[0].argmax()]
    plt.imshow(origin_img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()


train_dir = 'C:/Users/yjias/Desktop/10_food_classes_all_data/train/'
test_dir = 'C:/Users/yjias/Desktop/10_food_classes_all_data/test/'
data_dir = pathlib.Path(train_dir)
class_names = [item.name for item in data_dir.glob("*")]
#view_random_image(train_dir, 'sushi')

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen_augmented = ImageDataGenerator(rescale=1./255,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical')
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224, 224),
                                                                   batch_size=32,
                                                                   class_mode='categorical',
                                                                   shuffle=True)
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='categorical')
"""
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_1.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
history_1 = model_1.fit(train_data_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))

plot_loss_curves(history_1)
model_1.save("10_food_model.h5")
"""
model = tf.keras.models.load_model('10_food_model.h5')
test_pic = 'C:/Users/yjias/Desktop/test_img.png'
pred_and_plot(model, test_pic, class_names)

