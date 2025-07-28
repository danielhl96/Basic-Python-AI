from rcnn import rcnn
from image_ops import load_image
import cv2

img = load_image("00-924050_01__1200x1200.jpg",500,500)

result = rcnn(img)
print(result)
for elem in result:
    x1, y1, x2, y2 = elem['coords']
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 