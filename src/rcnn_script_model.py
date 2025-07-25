from re import X
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import readFile
import os
import argparse
import cv2
import imutils
import numpy as np

parser = argparse.ArgumentParser(description="Path of training and validation set")
parser.add_argument('-t', '--train', type=str, required=True, help="Path for training file")
parser.add_argument('-v', '--vali', type=str, required=True, help="Path for validation file")
parser.add_argument('-c', '--classification', type=bool, required=True, help="Flag for classification path")
parser.add_argument('-b', '--boundingbox', type=bool, required=True, help="Flag for boundingbox path")
args = parser.parse_args()



tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

def build_model(train_data, vali_data,bb,cl):
    coords_list, img_list, status_list = create_Data_List(train_data)
    coords_list_vali, img_list_vali, status_list_vali = create_Data_List(vali_data)

    train_images = np.array(img_list, dtype="float32") / 255.0
    train_coords = np.array(coords_list, dtype="float32")

    vali_images = np.array(img_list_vali, dtype="float32") / 255.0
    vali_coords = np.array(coords_list_vali, dtype="float32")

    trainTargets = {
        "cl_head": np.array(status_list),
        "bb_head": train_coords
    }
    valiTargets = {
        "cl_head": np.array(status_list_vali),
        "bb_head": vali_coords
    }

    input_shape = (500, 500, 3)
    input_layer = tf.keras.layers.Input(input_shape)

    base_layers = layers.experimental.preprocessing.Rescaling(1./255, name='bl_1')(input_layer)
    base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
    base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
    base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
    base_layers = layers.Flatten(name='bl_8')(base_layers)

    locator_branch = layers.Dense(64, activation='relu', name='bb_1')(base_layers)
    locator_branch = layers.Dense(32, activation='relu', name='bb_2')(locator_branch)
    locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)

    classifier_branch = layers.Dense(64, activation='relu', name='cl_1')(base_layers)
    classifier_branch = layers.Dense(2, activation='softmax', name='cl_head')(classifier_branch)


    # Nur einfrieren, wenn Modell geladen wurde:
    if os.path.exists("my_model.keras") and bb and not cl:
        model = load_model("my_model.keras")
        print("Train the classification path")
        # Nur Klassifikationszweig trainieren
        for layer in model.layers:
            layer.trainable = layer.name.startswith('cl_')
    elif os.path.exists("my_model.keras") and cl and not bb:
        model = load_model("my_model.keras")
        print("Train the bounding box path")
        for layer in model.layers:
            layer.trainable = layer.name.startswith('bb_')
    else:
        model = tf.keras.Model(input_layer, outputs=(locator_branch, classifier_branch))
        # Neues Modell → alles trainieren
        print("Train the complete model")
        for layer in model.layers:
            layer.trainable = True


    losses = {
        "bb_head": tf.keras.losses.MSE,
        "cl_head": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    }

    model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

    model.fit(train_images, trainTargets,
              validation_data=(vali_images, valiTargets),
              batch_size=4,
              epochs=5,
              verbose=1)

    print("[INFO] saving object detector model...")
    model.save("my_model.keras")


def read_data(train_dir,vali_dir):
    train_data = readFile.readFile(train_dir)
    vali_data = readFile.readFile(vali_dir)
    return train_data,vali_data

def create_Data_List(data):
    img_list = []
    coords_list = []
    status_list = []
    for i in range(0,len(data)):
        #if data[i]["status"] == 1:
            status_list.append(data[i]["status"])
            image = load_img(data[i]["name"], target_size=(500, 500))
            image = img_to_array(image)
            img_list.append(image)
            (h, w) = image.shape[:2]
            for j in range(0,len(data[i]["box"])):
                list = data[i]["box"][j]
                X = list[0]/w
                Y = list[1]/h
                x = list[2]/w
                y = list[3]/h
                coords_list.append((X,Y,x,y))
    return coords_list, img_list, status_list

print(f"Trainingsset: {args.train}")
print(f"Validierungsset: {args.vali}")

train_data,vali_data = read_data(args.train,args.vali)
build_model(train_data,vali_data,args.boundingbox,args.classification)