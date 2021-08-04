import cv2
img = cv2.imread('MUKAKU/Charles_Chang/Charles_Chang_0001.jpg')
resized = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
cv2.imshow('ImageWindow', resized)
cv2.waitKey()
cv2.destroyAllWindows()
