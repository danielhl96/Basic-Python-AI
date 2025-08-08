from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import cv2
import os
import gdown
from image_ops import crop_image, load_image

base_dir = os.path.dirname(os.path.abspath(__file__))  # Ordner, in dem rcnn.py liegt
model_path = os.path.join(base_dir, "my_model.keras")

if os.path.isfile(model_path):
    print("Model found at:", model_path)
    model = load_model(model_path)
else:
    print("Model not found. Downloading...")
    url = "https://drive.google.com/uc?id=1M8wJQ1mXQvl_xyU91YqFbQemiZNcpfjW"
    gdown.download(url, model_path)
    model = load_model(model_path)

def rcnn(img):

    print("Current working dir:", os.getcwd())
    print("DEBUG:")
    print("base_dir:", base_dir)
    print("model_path:", model_path)
    print("File exists:", os.path.isfile(model_path))

    #Create ROI`s `
    img_list = []
    
    image1 = crop_image(img,0,500,0,170)
    image2 =  crop_image(img,0,500,170,340)
    image3 = crop_image(img,0,500,340,500)

    blank_image1 = np.zeros((500,500,3), np.uint8)
    blank_image1.fill(255) 
    blank_image1[0:500,0:170] = image1

    blank_image2 = np.zeros((500,500,3), np.uint8)
    blank_image2.fill(255) 
    blank_image2[0:500,170:340] = image2

    blank_image3 = np.zeros((500,500,3), np.uint8)
    blank_image3.fill(255) 
    blank_image3[0:500,340:500] = image3

    img_list.append(blank_image1)
    img_list.append(blank_image2)
    img_list.append(blank_image3)

    #Calculate the prediction 
    list_cord = []
    confidences = []
    confidence_threshold = 0.5
    nms_threshold = 0.4
    result = []
    for elem in img_list:
        elem = img_to_array(elem) / 255.0
        elem = np.expand_dims(elem, axis=0)
        preds = model.predict(elem)
        confidence = float(np.max(preds[1][0]))
        confidences.append(float(confidence))
        startX, startY, endX, endY = preds[0][0]
        h = img.shape[0]
        w = img.shape[1]
        startX = int(startX*w)
        startY = int(startY*h)
        endX = int(endX*w)
        endY = int(endY*h)
        list_cord.append(((float(startX),float(startY),float(endX),float(endY))))
    indices = cv2.dnn.NMSBoxes(list_cord, confidences, confidence_threshold, nms_threshold)
    for elem in indices:
        d = {"coords": list_cord[elem],"confidence": confidences[elem]}
        result.append(d)
    return result