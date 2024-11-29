#!/bin/python3

import sys
import argparse
from time import sleep
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import time

import numpy as np
import cv2
from cv_bridge import CvBridge
import ament_index_python
import os

# Package includes
package_path = ament_index_python.get_package_share_directory("drone_behaviors")
sys.path.append(package_path)

from src.control_functions import band_midpoint, search_top_line, search_bottom_line, save_profiling, search_farthest_column
from droneExpertPilot import NUM_POINTS, SPACE

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
        self.timer = self.create_timer(5.0, self.save_data)

        # Create the profiling directory if it doesn't exist
        self.outDir = out_dir
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        self.traceBool = trace

    def save_data(self):
        save_profiling(self.outDir + '/profiling_image.txt', self.profiling)


    def show_trace(self, label, mono_img, original):
        # distancePoint = search_farthest_column(mono_img)
        # top_point = search_top_line(mono_img)
        # bottom_point = search_bottom_line(mono_img)


        # # Image to 3 channels
        # mono_img_color = cv2.merge([mono_img, mono_img, mono_img])

        # topPoint = search_top_line(mono_img)
        # bottomPoint = search_bottom_line(mono_img)
        # lineDiv = NUM_POINTS

        # error = 0
        # # self.get_logger().info("%d %d" % (bottomPoint, topPoint))
        # intervals = (bottomPoint - topPoint) / lineDiv
        # if int(intervals) != 0:
        #     for i in range(0, (bottomPoint - topPoint), int(intervals)):
        #         error = band_midpoint(mono_img, topPoint + i - SPACE , topPoint + i + 1)
        #         #self.get_logger().info("%d %d %d" % (error[0], error[1], i))


        #         # Draws the points in the original image
        #         cv2.circle(mono_img_color, (int(error[0]), int(i + top_point)), 5, (0, 255, 0), -1) 


        #     #cv2.circle(mono_img_color, (red_nearest[0], bottom_point), 5, (0, 255, 0), -1) 
        # cv2.circle(mono_img_color, (distancePoint, top_point), 5, (255, 0, 0), -1) 
        # Show the image
        cv2.imshow(label, mono_img)
        cv2.waitKey(1)

    def color_filter(self, image):

        # Apply a red filter to the image
        red_lower = np.array([0, 0, 70])
        red_upper = np.array([50, 50, 255])
        red_mask = cv2.inRange(image, red_lower, red_upper)

        return red_mask

    def image_aperture(self, mask):
        erosion_kernel = np.ones((2, 2), np.uint8)
        dilate_kernel = np.ones((10, 10), np.uint8)
        n_erosion = 1
        n_dilatation = 1

        # Perform aperture
        eroded_mask = cv2.erode(mask, erosion_kernel, iterations=n_erosion)
        dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=n_dilatation)

        return dilated_mask
    
    def filter_contours(self, contours):
        limit_area = 20
        
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
        initTime = time.time()
        red_mask = self.color_filter(cv_image)
        img = self.image_aperture(red_mask)

        # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filtered_contours = self.filter_contours(contours)
        # self.profiling.append(f"\nFilter time = {time.time() - initTime}")


        # # Draws only the big contour
        # initTime = time.time()
        # mono_img = np.zeros_like(img)

        # if filtered_contours != None:
        #     for contour in filtered_contours:
        #         cv2.drawContours(mono_img, [contour], -1, 255, thickness=cv2.FILLED)
        # self.profiling.append(f"\nDrawing time = {time.time() - initTime}")

        # Publish the filtered image
        msg = bridge.cv2_to_imgmsg(img, encoding="mono8")
        self.filteredPublisher_.publish(msg)

        self.profiling.append(f"\nCallback time = {time.time() - CallInitTime}")

        # Traces

        # Display the image with contours
        if self.traceBool:
            self.show_trace("Outline: ", img, img)


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