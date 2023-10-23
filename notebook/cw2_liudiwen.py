import cv2
import numpy as np
def detect_strongest_vertical_edge_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(laplacian)

    return max_loc if abs(max_val) > abs(min_val) else min_loc

def detect_vertical_edge(image):
    sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sobel_vertical)

    return max_loc if abs(max_val) > abs(min_val) else min_loc

def compute_velocity(point1, point2, pixel_size, time_interval):
    displacement = ((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)** 0.5

    displacement_meters = displacement * pixel_size

    velocity = displacement_meters / time_interval

    return displacement_meters, velocity

image1 = cv2.imread('/Users/tianze/cs4243_lab/CS4243_2023_images_small/fr11.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('/Users/tianze/cs4243_lab/CS4243_2023_images_small/fr12.png', cv2.IMREAD_GRAYSCALE)

point1 = detect_strongest_vertical_edge_laplacian(image1)
point2 = detect_strongest_vertical_edge_laplacian(image2)

cv2.arrowedLine(image1, point1, point2, (255, 0, 0), 2)

pixel_size = 0.05
time_interval = 2.5
displacement, velocity = compute_velocity(point1, point2, pixel_size, time_interval)

print(f"Displacement: {displacement} m")
print(f"Velocity of the object: {velocity} m/s")


cv2.imshow('Displacement Vector', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()



point1 = detect_vertical_edge(image1)
point2 = detect_vertical_edge(image2)

cv2.arrowedLine(image1, point1, point2, (255, 0, 0), 2)

pixel_size = 0.05
time_interval = 2.5
displacement, velocity = compute_velocity(point1, point2, pixel_size, time_interval)

print(f"Displacement: {displacement} m")
print(f"Velocity of the object: {velocity} m/s")


cv2.imshow('Displacement Vector', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()