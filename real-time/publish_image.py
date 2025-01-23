import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.publisher_ = self.create_publisher(Image, '/camera/black', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        # Initialize the video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 10)

        self.bridge = CvBridge()

    def calibrate(self, image, camera_matrix, distortion_coefficients):
        h, w = image.shape

        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))

        # self.get_logger().info(f"new matrix: {newcameramatrix}")

        dst = cv2.undistort(image, camera_matrix, distortion_coefficients, None, newcameramatrix)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

    def timer_callback(self):
        ret, frame = self.cap.read()

        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            camera_matrix = np.array([[547.9112, 0., 314.1736],
                                           [0., 561.7509, 253.7715],
                                           [0., 0., 1.]], dtype=np.float32)

            distortion_coef = np.array([-0.14725391, 0.59321798, 0.03298772, 0.01081673], dtype=np.float32)

            calibrated_img = self.calibrate(gray_frame, camera_matrix, distortion_coef)

            image_message = self.bridge.cv2_to_imgmsg(calibrated_img, encoding="mono8")
            # cv2.imwrite("./results/img_v2.png", calibrated_img)


            self.publisher_.publish(image_message)
            self.get_logger().info('Publishing image')


def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()

    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        camera_publisher.get_logger().info('Shutting down')
    finally:
        camera_publisher.cap.release()
        camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
