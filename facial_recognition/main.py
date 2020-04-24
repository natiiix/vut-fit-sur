import cv2
import numpy as np
import os
from random import shuffle

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D
from tensorflow.keras import optimizers

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import load_model

import time

TRAIN_DIR = "train_dir"
VALIDATION_DIR = "validation_dir"
TEST_DIR = "test_dir"
LOG_DIR = f"{int(time.time())}"

MODEL_NAME = "face_recognition.model"
LR = 1e-3

def label_img(img):
    return [1, 0] if img.startswith("m429") else [0, 1]

def create_data(d, name):
    data = list()

    for img in os.listdir(d):
        label = label_img(img)
        path = os.path.join(d, img)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        data.append([np.array(img), np.array(label)])
    
    shuffle(data)
    np.save(name, data)

    return data

def process_test_data():
    testing_data = list()

    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, img)
        name = img
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        testing_data.append([np.array(img), name])
    
    shuffle(testing_data)
    np.save("testing_data.npy", testing_data)

    return testing_data

def build_model_tflearn():
    convnet = input_data(shape = [None, 80, 80, 1], name = "input")

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    #convnet = conv_2d(convnet, 64, 5, activation='relu')
    #convnet = max_pool_2d(convnet, 5)

    #convnet = conv_2d(convnet, 128, 5, activation='relu')
    #convnet = max_pool_2d(convnet, 5)

    #convnet = conv_2d(convnet, 64, 5, activation='relu')
    #convnet = max_pool_2d(convnet, 5)

    #convnet = conv_2d(convnet, 32, 5, activation='relu')
    #convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model

def build_model_keras(hp):
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int("input_units", 16, 256, 16), (3, 3), input_shape=[80, 80, 1]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int("n_layers", 1, 8)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units", 16, 256, 16), (3, 3), input_shape=[80, 80, 1]))
        model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def load_model_tflearn():
    if os.path.exists(MODEL_NAME + ".meta"):
        model.load(MODEL_NAME)
        return model

def recognize(model, testing_data):
    for data in testing_data:
        img = data[0]
        img_index = data[1]

        data = img.reshape(-1, 80, 80, 1)

        model_output = model.predict([data])[0]
        print(str(img_index[:8]) + " " + str(1 - model_output[0]) + " " + str(1 if 1 - model_output[0] > 0.5 else 0))

if __name__ == "__main__":
    # prepare data
    #train_data = create_data(TRAIN_DIR, "train_data.npy")
    train_data = np.load("train_data.npy", allow_pickle=True)
        
    #validation_data = create_data(VALIDATION_DIR, "test_data.npy")
    validation_data = np.load("validation_data.npy", allow_pickle=True)

    #testing_data = process_test_data()
    testing_data = np.load("testing_data.npy", allow_pickle=True)

    # prepare training data to fit the model
    train_X = np.array([i[0] for i in train_data]).reshape(-1, 80, 80, 1)
    train_Y = [i[1] for i in train_data]

    # prepare validation data to fit into the model
    validation_X = np.array([i[0] for i in validation_data]).reshape(-1, 80, 80, 1)
    validation_Y = [i[1] for i in validation_data]

    train_Y = list()
    for i in train_data:
        train_Y.append(i[1][0])
    train_Y = np.asarray(train_Y)

    validation_Y = list()
    for i in validation_data:
        validation_Y.append(i[1][0])
    validation_Y = np.asarray(validation_Y)

    # get model tflearn
    #model = build_model_tflearn()
    #model = load_model_tflearn()

    # tflearn model training
    #model.fit({'input': train_X}, {'targets': train_Y}, n_epoch = 5, validation_set=({'input': validation_X}, {'targets': validation_Y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    # keras model tuning
    #keras_tuner = RandomSearch(
    #    build_model_keras,
    #    objective="val_acc",
    #    max_trials=70,
    #    executions_per_trial=3,
    #    directory=LOG_DIR
    #)

    #keras_tuner.search(x=train_X, y=train_Y, epochs=10, batch_size=16, validation_data=(validation_X, validation_Y))

    #keras_tuner.get_best_models()[0].save("best_model.h5")

    model = load_model("best_model.h5")
    recognize(model, testing_data)
