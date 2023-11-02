#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import ament_index_python
import torch
from torch.utils.data import DataLoader
import sys
from torchvision import transforms


package_path = ament_index_python.get_package_share_directory("drone_driver")
sys.path.append(package_path + "/include")
from models import pilotNet
from train import DATA_PATH, load_checkpoint
from data import dataset_transforms


# Limit velocitys
MAX_ANGULAR = 4.5 
MAX_LINEAR = 9 
MIN_LINEAR = 1

class carController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.imageSubscription = self.create_subscription(Image, '/cam_f1_left/image_raw', self.listener_callback, 10)
        self.img = None

        # Neural network
        self.model = pilotNet()
        load_checkpoint(self.model)

        self.device = torch.device("cuda:0")
        self.model.to(self.device)


    def listener_callback(self, msg):
        bridge = CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, "bgr8")
        img_tensor = dataset_transforms(self.img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        # Image inference
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Velocity set
        vel = predictions[0].tolist()
        vel_msg = Twist()
        vel_msg.linear.x = float(vel[0])
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = float(vel[1])

        print(float(vel[0]), float(vel[1]))

        self.publisher_.publish(vel_msg)

    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = carController()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()