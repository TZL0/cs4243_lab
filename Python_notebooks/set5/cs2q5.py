import cv2
import numpy as np


frame1 = cv2.imread("/Users/tianze/cs4243_lab/CS4243_2023_images_small/fr11.png", 0)
frame2 = cv2.imread("/Users/tianze/cs4243_lab/CS4243_2023_images_small/fr12.png", 0)

x_sobel_kernel = np.array([[-1 , 0 , 1] , [-2 , 0 , 2] , [-1 , 0 , 1]])

sobel_x1 = cv2.filter2D(src=frame1, ddepth=-1, kernel=x_sobel_kernel)
sobel_x2 = cv2.filter2D(src=frame2, ddepth=-1, kernel=x_sobel_kernel)


y1, x1 = np.unravel_index(np.argmax(sobel_x1), sobel_x1.shape)
y2, x2 = np.unravel_index(np.argmax(sobel_x2), sobel_x2.shape)


cv2.namedWindow("fr11", cv2.WINDOW_NORMAL)
cv2.imshow("fr11", sobel_x1)
cv2.waitKey(0)
cv2.imshow("fr12", sobel_x2)
cv2.waitKey(0)
merged = cv2.addWeighted(sobel_x1, 0.5, sobel_x2, 0.5, 0)
cv2.arrowedLine(merged, (x1, y1), (x2, y2), (255, 255, 255), 1)
cv2.imshow("fr11+fr12", merged)
cv2.waitKey(0)

d_pixels = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
d_meters = d_pixels * 0.05
v = d_meters / 2.5

print("Displacement:", d_meters, 'm')
print("Velocity:", v, "m/s")

# Real-world applications of this algorithm:
# j. This algorithm can be used in scenarios like traffic monitoring to detect the speed of vehicles, in sports analytics to compute the speed of a ball or player, or in security applications to detect suspicious movements.
# k. The real-world applications include video surveillance systems to detect fast-moving objects, autonomous vehicles for obstacle detection and velocity estimation, robotics for motion planning, and in augmented reality to understand and respond to real-world movements.
