import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random

def vis_prediction(model, test_set):
    preds = model.predict(test_set/255)
    plt.figure(figsize=(10, 10))
    for i in range(4):
        rand_index = random.choice(range(len(test_set)))
        plt.subplot(2, 2, i+1)
        plt.imshow(test_set[rand_index], cmap=plt.cm.binary)
        poss = preds[rand_index]
        max_poss = np.argmax(poss)
        plt.title(class_names[max_poss])
        plt.axis(False)
    plt.show()

def vis_by_type(type_name, x_set, label_set, model):
    if type_name in class_names:
        type_ind = class_names.index(type_name)
        x_set_type = []
        for i in range(len(x_set)):
            if label_set[i] == type_ind:
                x_set_type.append(x_set[i])
        print(len(x_set_type))

        rand_index = random.choice(range(len(x_set_type)))
        print(x_set_type[rand_index].shape)
        preds = model.predict(x_set_type[rand_index]/255)
        plt.figure(figsize=(7,7))
        plt.imshow(x_set_type[rand_index], cmap=plt.cm.binary)
        max_poss = np.argmax(preds)
        plt.title(class_names[max_poss])
        plt.show()

    else:
        print("We don't have this type of clothes")


(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print(f"Training sample: \n{train_data.shape}\n")
#print(f"Training label: {np.max(train_labels)}")
#plt.figure(figsize=(7, 7))
#plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
#plt.title(class_names[train_labels[rand_index]])
#plt.show()

tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])

#norm_history = model_1.fit(train_data/255, train_labels,
#                                epochs=20,
#                                validation_data=(test_data/255, test_labels))
#model_1.save('fashion_mnist_model1.h5')
model_1 = tf.keras.models.load_model('fashion_mnist_model1.h5')
preds = model_1.predict(test_data/255)

#vis_prediction(model_1, test_data)
vis_by_type('Dress', train_data, train_labels, model_1)
