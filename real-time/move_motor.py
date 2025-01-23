import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random
import serial
import time


class MotorSpeedController(Node):
    def __init__(self):
        super().__init__('motor_speed_controller')
        self.ser = serial.Serial("/dev/ttyUSB0", 57600, timeout=2)
        time.sleep(3)  # Wait for Arduino to initialize

        self.i = 0
        self.speeds = {
            0: (0, 0),
            1: (65, 65),
            10: (67, 55),
            19: (65, 65),
            26: (67, 55),
            35: (0, 0)
        }
        self.current_speed = (0, 0)

        self.motor_left_pub = self.create_publisher(Float32, '/motor_left_speed', 10)
        self.motor_right_pub = self.create_publisher(Float32, '/motor_right_speed', 10)

        self.timer = self.create_timer(1.0, self.update_motor_speeds)

    def send_data(self, data):
        self.ser.write(data.encode())

    def receive_data(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8').strip()
            return data
        return None

    def update_speed(self, left, right):
        self.send_data(f"m {left} {right}\r")
        response = self.receive_data()
        if response:
            self.get_logger().info(f"Message received: {response}")

        self.motor_left_pub.publish(Float32(data=float(left)))
        self.motor_right_pub.publish(Float32(data=float(right)))
        self.get_logger().info(f'Left Motor Speed: {left}, Right Motor Speed: {right}')

    def update_motor_speeds(self):
        self.current_speed = self.speeds.get(self.i, self.current_speed)

        left, right = self.current_speed

        self.update_speed(left, right)

        self.i += 1


if __name__ == '__main__':
    rclpy.init()

    motor_speed_controller = MotorSpeedController()

    try:
        rclpy.spin(motor_speed_controller)
    except KeyboardInterrupt:
        pass
    finally:
        motor_speed_controller.ser.close()
        motor_speed_controller.destroy_node()
        rclpy.shutdown()
