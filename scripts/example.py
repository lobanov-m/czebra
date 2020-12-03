import cv2
import czebra as cz

model = cz.load_model("mxnet_bisenet_egolane")

img = cv2.imread('../resources/1.jpg')

result = model.predict(img)
result.segmentation.visualize_segmentation(img)

cv2.imshow('', img)
cv2.waitKey()
