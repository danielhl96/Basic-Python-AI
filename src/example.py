from rcnn import rcnn
from image_ops import load_image

img = load_image("example.jpeg",500,500)
result = rcnn(img)
print(result)

