import cv2
import numpy as np
import os
from random import shuffle

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = "train_dir"
VALIDATION_DIR = "validation_dir"
TEST_DIR = "test_dir"

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

    i = 0
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_index = i

        testing_data.append([np.array(img), img_index])
        i += 1
    
    shuffle(testing_data)
    np.save("testing_data.npy", testing_data)

    return testing_data

def build_model_tflearn():
    convnet = input_data(shape = [None, 80, 80, 1], name = "input")

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model

def load_model():
    if os.path.exists(MODEL_NAME + ".meta"):
        model.load(MODEL_NAME)
        return model

def recognize(model, testing_data):
    for data in testing_data:
        img = data[0]
        img_index = data[1]

        data = img.reshape(80, 80, 1)

        model_output = model.predict([data])[0]
        print(str(img_index) + ": " + str(model_output[0]))

if __name__ == "__main__":
    #train_data = create_data(TRAIN_DIR, "train_data.npy")
    train_data = np.load("train_data.npy", allow_pickle=True)
        
    #validation_data = create_data(VALIDATION_DIR, "test_data.npy")
    validation_data = np.load("validation_data.npy", allow_pickle=True)

    #testing_data = process_test_data()
    #testing_data = np.load("testing_data.npy", allow_pickle=True)

    model = build_model_tflearn()
    model = load_model()

    train_X = np.array([i[0] for i in train_data]).reshape(-1, 80, 80, 1)
    train_Y = [i[1] for i in train_data]

    validation_X = np.array([i[0] for i in validation_data]).reshape(-1, 80, 80, 1)
    validation_Y = [i[1] for i in validation_data]

    model.fit({'input': train_X}, {'targets': train_Y}, n_epoch = 3, validation_set=({'input': validation_X}, {'targets': validation_Y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

    #recognize(model, testing_data)
