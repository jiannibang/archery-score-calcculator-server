import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# k聚类
# img = cv2.imread(args["image"])
# Z = img.reshape((-1,3))

# # convert to np.float32
# Z = np.float32(Z)

# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 5
# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))

# cv2.imshow('res2',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 找圆
# img = cv2.imread(args["image"],0)
# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,1,
#                             param1=50,param2=200,minRadius=10,maxRadius=200)

# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 找线
# img = cv2.imread(args["image"],0)
# img = cv2.GaussianBlur(img,(5,5),0)
# gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# minLineLength = 0
# maxLineGap = 1
# lines = cv2.HoughLinesP(edges,1,np.pi/180/,100,minLineLength,maxLineGap)
# for line in lines:
#   for x1,y1,x2,y2 in line:
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# cv2.imshow('detected circles',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread(args["image"],0)
blurred = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blurred,100,200)
blurredEdges = cv2.GaussianBlur(edges,(5,5),0)

cimg = cv2.cvtColor(blurredEdges,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(blurredEdges,cv2.HOUGH_GRADIENT,1,50,
                            param1=50,param2=150,minRadius=200,maxRadius=250)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    for num in range(1,9):
       cv2.circle(cimg,(i[0],i[1]),int(i[2]*num/9),(0,0,255),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()