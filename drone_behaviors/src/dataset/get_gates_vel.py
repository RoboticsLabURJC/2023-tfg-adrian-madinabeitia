#!/bin/python3

#! This node works with the the following repo 
#! https://github.com/Adrimapo/project_crazyflie_gates
#! to obtain the drone velocity's for the dataset


import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import signal
import rosbag2_py
import argparse
import os
from rclpy.serialization import serialize_message
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image


class velPublisher(Node):
    def sigint_handler(*self, args):
        print('Exiting...')
        self.writer = None

        self.destroy_node()
        rclpy.shutdown()


    def __init__(self, profiling_directory: str = '.'):
        super().__init__('vel_publisher')

        self.vel_topic = 'commanded_vels'
        self.imageTopic = '/cf0/sensor_measurements/hd_camera/image_raw'
        self.tfTopic = "/tf"


        self.velPublisher_ = self.create_publisher(Point, self.vel_topic, 10)
        self.timer = self.create_timer(0.1, self.timer_vel_publisher)

        self.subscription = self.create_subscription( Twist, '/gz/cf0/cmd_vel', self.listener_callback, 10)
        self.imageSubscription = self.create_subscription(Image, self.imageTopic, self.image_callback, 1)
        self.tfSubscription = self.create_subscription(TFMessage, self.tfTopic, self.tf_callback, 1)

        self.linearVel = 0.0
        self.linearVelRaw = 0.0
        self.angularVel = 0.0

        self.recordId = 0
        self.recording = False

        # Create the profiling directory if it doesn't exist
        self.profilingDir = profiling_directory
        if not os.path.exists(self.profilingDir):
            os.makedirs(self.profilingDir)
        
        self.writer = None
        self.open_bag()

    def image_callback(self, msg):
        # Saves the topic if desired
        if self.recording:
            self.writer.write(
                self.imageTopic,
                serialize_message(msg),
                self.get_clock().now().nanoseconds)    

    # Used for recording rosbags
    def tf_callback(self, msg):
        if self.recording:
            self.writer.write(
                self.tfTopic,
                serialize_message(msg),
                self.get_clock().now().nanoseconds)
            
    def timer_vel_publisher(self):
        msg = Point()
        umbral = 0.8

        # If the drone is quiet we dont need the data (in this application)
        if self.linearVel > umbral or self.angularVel > umbral:
            # Min vel reached 
            self.recording = True

            # Publish the info for training
            msg.x = float(self.linearVel)
            msg.y = float(self.linearVelRaw)
            msg.z = float(self.angularVel)
            self.velPublisher_.publish(msg)
        
            # Saves the rosbag
            self.writer.write(
                self.vel_topic,
                serialize_message(msg),
                self.get_clock().now().nanoseconds)  
        else:
            self.recording = False
            

    def listener_callback(self, msg):
        linearX = msg.linear.x
        linearY = msg.linear.y

        self.angularVel = msg.angular.z
        
        # We dont have filter in this implementation
        self.linearVelRaw = 0
        
        # Gets the real linear vel
        self.linearVel = linearX * math.cos(self.angularVel) - linearY * math.sin(self.angularVel)
    

    def open_bag(self):
        ## Creates the writer
        self.writer = rosbag2_py.SequentialWriter()
        self.converter_options = rosbag2_py._storage.ConverterOptions('', '')

        filePath = self.profilingDir + '/rosbag' + str(self.recordId)
        # Ensures that is a new directory
        while os.path.exists(filePath):
            self.recordId += 1
            filePath = self.profilingDir + '/rosbag' + str(self.recordId)

        storage_options = rosbag2_py._storage.StorageOptions(
            uri=filePath,
            storage_id='sqlite3')
        
        self.writer.open(storage_options, self.converter_options)

        # Saves the desired topics
        ## Velocity topic
        vel_topic_info = rosbag2_py._storage.TopicMetadata(
            name=self.vel_topic,
            type='geometry_msgs/msg/Point',
            serialization_format='cdr')

        ## Image topic
        image_topic_info = rosbag2_py._storage.TopicMetadata(
            name=self.imageTopic,
            type='sensor_msgs/msg/Image',
            serialization_format='cdr')

        ## Tf topic
        tf_topic_info = rosbag2_py._storage.TopicMetadata(
            name=self.tfTopic,
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr')

        self.writer.create_topic(vel_topic_info)
        self.writer.create_topic(image_topic_info)
        self.writer.create_topic(tf_topic_info) 
        self.recordId += 1

        return filePath

def main(args=None):
    rclpy.init(args=args)

    # Gets the necessary arguments
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output_directory', type=str, help='Directory to save profiling files', required=True)
    parsed_args, _ = parser.parse_known_args()
    
    vel_publisher = velPublisher(profiling_directory=parsed_args.output_directory)
    signal.signal(signal.SIGINT, vel_publisher.sigint_handler)
    rclpy.spin(vel_publisher)

    # End of execution
    vel_publisher.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()