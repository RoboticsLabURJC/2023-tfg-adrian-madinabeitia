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

from control_functions import band_midpoint, search_top_line, search_bottom_line
from droneExpertPilot import LIMIT_UMBRAL



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




class droneController(Node):

    def __init__(self):
        super().__init__('drone_line_follow')
        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 1)
        self.imageSubscription = self.create_subscription(Image, '/drone0/sensor_measurements/frontal_camera/image_raw', self.listener_callback, 1)

        self.px_rang = MAX_PIXEL - MIN_PIXEL

    def show_trace(self, rgb_image, original):
        height, width, channels = rgb_image.shape

        top_point = search_top_line(rgb_image)
        bottom_point = search_bottom_line(rgb_image)

        red_farest = band_midpoint(rgb_image, top_point, top_point+LIMIT_UMBRAL)
        red_nearest = band_midpoint(rgb_image, bottom_point-LIMIT_UMBRAL, bottom_point)

        cv2.circle(rgb_image, red_nearest, RADIUS, TRACE_COLOR, RADIUS)
        cv2.circle(rgb_image, red_farest, RADIUS, TRACE_COLOR, RADIUS)
        cv2.line(rgb_image, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

        cv2.imshow("Dilated mask", rgb_image)
        cv2.imshow("Original", original)
        cv2.waitKey(1)

    def listener_callback(self, msg):
        erosion_kernel = np.ones((1, 1), np.uint8)
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
        msg = bridge.cv2_to_imgmsg(dilated_mask, encoding="mono8")
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
