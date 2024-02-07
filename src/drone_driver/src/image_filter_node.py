#!/bin/python3

import sys

from time import sleep
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import time

import numpy as np
import cv2
from cv_bridge import CvBridge
import ament_index_python

# Package includes
package_path = ament_index_python.get_package_share_directory("drone_driver")
sys.path.append(package_path)

from include.control_functions import band_midpoint, search_top_line, search_bottom_line, save_profiling, search_farest_column
from droneExpertPilot import BOTTOM_LIMIT_UMBRAL, UPPER_LIMIT_UMBRAL



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


class imageFilterNode(Node):

    def __init__(self):
        super().__init__('drone_line_follow')
        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 1)
        self.imageSubscription = self.create_subscription(Image, '/drone0/sensor_measurements/frontal_camera/image_raw', self.listener_callback, 1)

        self.px_rang = MAX_PIXEL - MIN_PIXEL
        self.profiling = []
        self.timer = self.create_timer(5.0, self.save_data)

    def save_data(self):
        save_profiling('./profiling_image.txt', self.profiling)


    def show_trace(self, label, mono_img, original):
        distancePoint = search_farest_column(mono_img)
        top_point = search_top_line(mono_img)
        bottom_point = search_bottom_line(mono_img)

        red_farest = band_midpoint(mono_img, top_point, top_point + UPPER_LIMIT_UMBRAL)
        red_nearest = band_midpoint(mono_img, bottom_point - BOTTOM_LIMIT_UMBRAL, bottom_point)

        # Convertir la imagen monocromática a 3 canales
        mono_img_color = cv2.merge([mono_img, mono_img, mono_img])

        # Dibujar los puntos en verde en la imagen original
        cv2.circle(mono_img_color, (red_farest[0], top_point), 5, (0, 255, 0), -1)  # Punto farest en verde
        cv2.circle(mono_img_color, (red_nearest[0], bottom_point), 5, (0, 255, 0), -1)  # Punto nearest en verde
        cv2.circle(mono_img_color, (distancePoint, top_point), 5, (255, 0, 0), -1)  # Punto farest en verde
        # Mostrar la imagen original con los puntos verdes
        cv2.imshow(label, mono_img_color)
        cv2.waitKey(1)

    def color_filter(self, image):

        # Apply a red filter to the image
        red_lower = np.array([0, 0, 100])
        red_upper = np.array([50, 50, 255])
        red_mask = cv2.inRange(image, red_lower, red_upper)

        return red_mask

    def image_aperture(self, mask):
        erosion_kernel = np.ones((2, 2), np.uint8)
        dilate_kernel = np.ones((7, 7), np.uint8)
        n_erosion = 1
        n_dilatation = 1

        # Perform aperture
        eroded_mask = cv2.erode(mask, erosion_kernel, iterations=n_erosion)
        dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=n_dilatation)

        return dilated_mask
    
    def filter_contours(self, contours):
        limit_area = 100
        
        # Filters small contours
        big_contours = [contour for contour in contours if cv2.contourArea(contour) > limit_area]

        if len(big_contours) != 0:
            # Find the contour with the maximum area
            max_contour = max(big_contours, key=cv2.contourArea)
            return [max_contour]
        else:
            return None



    def listener_callback(self, msg):

        CallinitTime = time.time()
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Filters
        initTime = time.time()
        red_mask = self.color_filter(cv_image)
        img = self.image_aperture(red_mask)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_contours(contours)
        self.profiling.append(f"\nFilter time = {time.time() - initTime}")


        # Draws tonly the big contour
        initTime = time.time()
        mono_img = np.zeros_like(img)

        if filtered_contours != None:
            for contour in filtered_contours:
                cv2.drawContours(mono_img, [contour], -1, 255, thickness=cv2.FILLED)
        self.profiling.append(f"\nDrawing time = {time.time() - initTime}")

        # Publish the filtered image
        msg = bridge.cv2_to_imgmsg(mono_img, encoding="mono8")
        self.filteredPublisher_.publish(msg)

        self.profiling.append(f"\nCallback time = {time.time() - CallinitTime}")

        # Traces

        # Display the image with contours
        # self.show_trace("Countors: ", mono_img, img)



if __name__ == '__main__':

    rclpy.init()

    img = imageFilterNode()
    
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
