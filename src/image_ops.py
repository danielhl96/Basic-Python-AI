import cv2
import numpy as np

c = -1
d = -1
ix = -1
iy = -1
drawing = False

def load_image(path,w,h):
    if w and h >= 0:
        return cv2.resize(cv2.imread(path), (w,h), interpolation = cv2.INTER_AREA)
    return -1

def crop_image(image,h1,h2,w1,w2):
    if (h1>=0 and w1>=0 and h2>=0 and w2>=0) and (image is not None):
        return image[h1:h2,w1:w2]
    return -1

def sharpen_image(image):
    if image is not None:
        kernel = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    return -1

def zoom(zoom_factor, image):
    for i in range(0,2):
        image = np.repeat(image,zoom_factor,axis=i)
    return image

def draw_rectangle_with_drag(event, x, y, flags, param):
  
    global ix, iy,drawing,img,c,d
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y            
              
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = copyImg.copy()
            cv2.rectangle(img, pt1 =(ix, iy), pt2 =(x, y),color =(0, 255, 255),thickness =1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img, pt1 =(ix, iy), pt2 =(x, y),color =(0, 255, 255),thickness =1)
        drawing = False
        c = x
        d = y