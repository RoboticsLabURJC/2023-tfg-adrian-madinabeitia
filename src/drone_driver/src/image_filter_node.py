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
LIMIT_UMBRAL = 15

class droneController(Node):

    def __init__(self):
        super().__init__('drone_line_follow')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 100)

        self.imageSubscription = self.create_subscription(Image, '/drone0/sensor_measurements/frontal_camera/image_raw', self.listener_callback, 10)

        self.px_rang = MAX_PIXEL - MIN_PIXEL

    def farest_point(self, image):
        img_width = image.shape[1]
        height_mid = int(image.shape[0] / 2)
            
        x = 0
        y = 0
        count = 0
            
        for row in range (height_mid, height_mid + LIMIT_UMBRAL):
            for col in range (img_width):
                    
                comparison = image[row][col] == np.array([0, 0, 0])
                if not comparison.all():
                    y += row
                    x += col 
                    count += 1
            
        if (count == 0):
            return (0, 0)

        return [int(x / count), int(y / count)]

    def draw_traces(self, image):
        img_width = image.shape[1]
        img_height = image.shape[0]
            
        for row_index in range(img_height):
            image[row_index][int(img_width / 2)] = np.array(TRACE_COLOR)
        
    def listener_callback(self, msg):

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            
        # Apply a red filter to the image
        red_lower = np.array([0, 0, 100])  
        red_upper = np.array([100, 100, 255]) 
        red_mask = cv2.inRange(cv_image, red_lower, red_upper)
        self.filteredImage = cv2.bitwise_and(cv_image, cv_image, mask=red_mask)

        cv2.imshow("img", self.filteredImage)
        cv2.waitKey(1) 




if __name__ == '__main__':

    rclpy.init()

    img = droneController()
    
    rclpy.spin(img)

    img.destroy_node()
    rclpy.shutdown()

    print("Clean exit")
    exit(0)
