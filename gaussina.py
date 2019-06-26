import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('5.jpg',0)
kernel = np.ones((5,5), np.uint8)

pup_blur = cv2.GaussianBlur(img, (55, 55), 0)

im_canny = cv2.Canny(pup_blur,5,17)
im_dilate = cv2.dilate(im_canny,kernel)



contours , hierarchyIndexes = cv2.findContours(im_dilate.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imwrite("result.jpg", img) 
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(im_canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([]) 
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows