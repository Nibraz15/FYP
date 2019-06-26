import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2.jpg',0)
kernel = np.ones((5,5), np.uint8)

pup_blur = cv2.GaussianBlur(img, (11, 11), 0)
pup_blur1 = cv2.GaussianBlur(img, (11, 11), 0)
pup_blur2 = cv2.GaussianBlur(img, (11, 11), 0)

im_canny = cv2.Canny(pup_blur,10,20)
im_dilate = cv2.dilate(im_canny,kernel)
laplacian = cv2.Laplacian(pup_blur1,cv2.CV_64F)
sobelx = cv2.Sobel(pup_blur2,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(pup_blur2,cv2.CV_64F,0,1,ksize=5)  # y


contours , hierarchyIndexes = cv2.findContours(im_dilate.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imwrite("result.jpg", img) 
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(im_canny,cmap = 'gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2,2,3),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows