import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

import cv2
import numpy as np

from real_time_odometry import LightGlueOdometryReal

import sys

class BlackImageSubscriber(Node):
    def __init__(self):
        super().__init__('black_image_subscriber')
        self.publisher = self.create_publisher(Odometry, '/odom', 1)
        # qos_profile = QoSProfile(
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     depth=10,
        #     reliability=QoSReliabilityPolicy.BEST_EFFORT
        # )

        self.subscription = self.create_subscription(
            Image,
            '/camera/black',
            self.perform_odom,
            1
        )
        self.subscription
        self.bridge = CvBridge()

        self.odom = LightGlueOdometryReal()
        self.latest_image = None
        self.get_logger().info('started working.')

    def perform_odom(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info(f"image dimension{cv_image.shape}")

            result = self.odom.evaluate(cv_image)

            self.publisher.publish(result)
            self.get_logger().info(
                f'Published Odometry: Position ({result.pose.pose.position}), Orientation ({result.pose.pose.orientation})')

            # cv2.imwrite("./results/img.png", cv_image)


            if np.count_nonzero(cv_image) == 0:
                self.get_logger().info('Received a black image.')

            else:
                self.get_logger().info('Received a non-black image.')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    black_image_subscriber = BlackImageSubscriber()

    try:
        rclpy.spin(black_image_subscriber)
        # rclpy.spin(black_image_subscriber.odom)
        # black_image_subscriber.perform_odom()
    except KeyboardInterrupt:
        pass
    finally:
        # black_image_subscriber.odom.destroy_node()
        black_image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
