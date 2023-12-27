#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, SensorDataQoS
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rosbag2_py import write_rosbag2


class SyncAndRecordNode(Node):
    def __init__(self):
        super().__init__('sync_and_record_node')

        self.twist_subscription = self.create_subscription(
            Twist,
            '/drone0/motion_reference/twist',
            self.twist_callback,
            QoSProfile(history=1)
        )

        self.image_subscription = self.create_subscription(
            Image,
            '/drone0/sensor_measurements/frontal_camera/image_raw',
            self.image_callback,
            SensorDataQoS()
        )

        self.bag_filename = 'synced_data.db3'
        self.bag_writer = write_rosbag2.PyRosbag2()

    def twist_callback(self, msg):
        # Aquí puedes procesar o guardar los mensajes Twist según tus necesidades
        print(f"Twist timestamp: {msg.header.stamp}")

    def image_callback(self, msg):
        # Aquí puedes procesar o guardar los mensajes de imagen según tus necesidades
        print(f"Image timestamp: {msg.header.stamp}")

    def save_to_rosbag(self):
        # Abre la bolsa ROS para escritura
        self.bag_writer.open(self.bag_filename, 'w')

        try:
            # Inicia el bucle de eventos
            rclpy.spin_once(self)

        except KeyboardInterrupt:
            pass

        finally:
            # Cierra la bolsa ROS después de procesar todos los mensajes
            self.bag_writer.close()


def main(args=None):
    rclpy.init(args=args)

    node = SyncAndRecordNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        # Inicia el bucle principal
        executor.spin()

    finally:
        # Cierra todo limpiamente
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

