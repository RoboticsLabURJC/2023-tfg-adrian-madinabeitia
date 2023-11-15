#!/bin/python3

from time import sleep
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np
import cv2
from cv_bridge import CvBridge

MIN_PIXEL = -360
MAX_PIXEL = 360

# Red filter parameters
RED_VALUE = 217
GREEN_VALUE = 41
BLUE_VALUE = 41

MAX_RANGE = 20
MIN_RANGE = 10

# Trace parameters
TRACE_COLOR = [0, 255, 0]
RADIUS = 2

# Distance to search the red line end
LIMIT_UMBRAL = 20


MAX_ANGULAR = 2

def search_top_line(image):
    img_height = image.shape[0]
    first_nonzero_row = 0

    for row in range(img_height):
        if np.any(image[row] != 0):
            first_nonzero_row = row
            break
    
    return first_nonzero_row

def band_midpoint(image, topw, bottomW):
    img_width = image.shape[1]
    img_height = image.shape[0]

    x = 0
    y = 0
    count = 0

    # Asegurarse de que el límite no exceda el tamaño de la imagen
    limit_umbral = min(LIMIT_UMBRAL, img_height - topw)
    limit = min(topw + limit_umbral, img_height-1)

    # Checks the image limits
    init = max(topw, 0)
    end = min(bottomW, img_height-1)

    for row in range(init, end):
        for col in range(img_width):

            comparison = image[row][col] != np.array([0, 0, 0])
            if comparison.all():
                y += row
                x += col 
                count += 1

    if count == 0:
        return (0, 0)

    return [int(x / count), int(y / count)]


class droneController(Node):

    def __init__(self):
        super().__init__('drone_line_follow')
        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 10)
        self.imageSubscription = self.create_subscription(Image, '/drone0/sensor_measurements/frontal_camera/image_raw', self.listener_callback, 10)

        self.px_rang = MAX_PIXEL - MIN_PIXEL

    def show_trace(self, rgb_image, original):
        height, width, channels = rgb_image.shape

        top_point = search_top_line(rgb_image)
        red_farest = band_midpoint(rgb_image, top_point, top_point+LIMIT_UMBRAL)
        red_nearest = band_midpoint(rgb_image, height-LIMIT_UMBRAL, height)

        cv2.circle(rgb_image, red_nearest, RADIUS, TRACE_COLOR, RADIUS)
        cv2.circle(rgb_image, red_farest, RADIUS, TRACE_COLOR, RADIUS)
        cv2.line(rgb_image, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

        cv2.imshow("Dilated mask", rgb_image)
        cv2.imshow("Original", original)
        cv2.waitKey(1)

    def listener_callback(self, msg):
        erosion_kernel = np.ones((2, 2), np.uint8)
        dilate_kernel = np.ones((6, 6), np.uint8)
        n_erosion = 1
        n_dilatation = 1

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Apply a red filter to the image
        red_lower = np.array([0, 0, 100])
        red_upper = np.array([50, 50, 255])
        red_mask = cv2.inRange(cv_image, red_lower, red_upper)
        
        # Perform aperture
        eroded_mask = cv2.erode(red_mask, erosion_kernel, iterations=n_erosion)
        dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=n_dilatation)
        rgb_image = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2RGB)

        # Publish the filtered image
        msg = bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        self.filteredPublisher_.publish(msg)
        
        self.show_trace(rgb_image, cv_image)



if __name__ == '__main__':

    rclpy.init()

    img = droneController()
    
    try:
        rclpy.spin(img) 
    except KeyboardInterrupt:
        print("Interrupted by user")

    img.destroy_node()
    try:
        rclpy.shutdown()
        print("Clean exit")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    exit()
