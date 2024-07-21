#!/bin/python3

import sys
import argparse
from time import sleep
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import time

import numpy as np
import math
import cv2
from cv_bridge import CvBridge
import ament_index_python
import os



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

    def __init__(self, out_dir=".", trace=True):
        super().__init__('drone_line_follow')
        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 1)
        self.imageSubscription = self.create_subscription(Image, '/drone0/sensor_measurements/frontal_camera/image_raw', self.listener_callback, 1)

        self.px_rang = MAX_PIXEL - MIN_PIXEL
        self.profiling = []
        #self.timer = self.create_timer(5.0, self.save_data)

        # Create the profiling directory if it doesn't exist
        self.outDir = out_dir
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        self.traceBool = trace

    # def save_data(self):
    #     save_profiling(self.outDir + '/profiling_image.txt', self.profiling)


    def show_trace(self, label, mono_img):

        cv2.imshow(label, mono_img)
        cv2.waitKey(1)

    def color_filter(self, image, color):

        # Apply a red filter to the image
        if color == "Red":
            lower = np.array([0, 0, 70])
            upper = np.array([50, 50, 255])
            mask = cv2.inRange(image, lower, upper)
        
        else:
            lower = np.array([50, 50, 50])
            upper = np.array([120, 120, 120])
            mask = cv2.inRange(image, lower, upper)

        return mask

    def image_aperture(self, mask):
        erosion_kernel = np.ones((5, 5), np.uint8)
        dilate_kernel = np.ones((10, 10), np.uint8)
        n_erosion = 1
        n_dilatation = 1

        # Perform aperture
        eroded_mask = cv2.erode(mask, erosion_kernel, iterations=n_erosion)
        dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=n_dilatation)


        return dilated_mask
    
    def filter_contours(self, contours):
        limit_area = 40
        contours, _ = cv2.findContours(contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filters small contours
        big_contours = [contour for contour in contours if cv2.contourArea(contour) > limit_area]

        if len(big_contours) != 0:
            # Find the contour with the maximum area
            max_contour = max(big_contours, key=cv2.contourArea)
            return [max_contour]
        else:
            return None


    def listener_callback(self, msg):

        CallInitTime = time.time()
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Filters
        red_mask = self.color_filter(cv_image, "G")
        aperture = self.image_aperture(red_mask)
        a = np.zeros_like(aperture)
        #contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_contours(a)


        # Draws only the big contour
        mono_img = np.zeros_like(aperture)

        if filtered_contours != None:
            for contour in filtered_contours:
                cv2.drawContours(mono_img, [contour], -1, 255, thickness=10)

        # Publish the filtered image
        msg = bridge.cv2_to_imgmsg(aperture, encoding="mono8")
        #msg = bridge.cv2_to_imgmsg(mono_img, encoding="mono8")
        self.filteredPublisher_.publish(msg)

        self.profiling.append(f"\nCallback time = {time.time() - CallInitTime}")

        # Traces

        # Display the image with contours
        if self.traceBool:
            self.show_trace("Outline: ", aperture)


def main(args=None):
    rclpy.init(args=args)
    # Gets the necessary arguments
    parser = argparse.ArgumentParser(description='Drone Controller with Profiling', allow_abbrev=False)
    parser.add_argument('--output_directory', type=str, help='Directory to save profiling files', required=True)
    parser.add_argument('--trace', type=str, help='Show the traces')
    parsed_args, _ = parser.parse_known_args()

    if parsed_args.trace == "True" or parsed_args.trace == "true":
        traceBool = True
    else:
        traceBool = False

    # Use the boolean directly, no need to convert to string
    img = imageFilterNode(out_dir=parsed_args.output_directory, trace=traceBool)
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


if __name__ == '__main__':
    main()