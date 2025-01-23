from datetime import datetime
from pathlib import Path
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

import torch
import numpy as np

from LightGlue.lightglue import SuperPoint, LightGlue, match_pair
from Tartanvo.Datasets.utils import dataset_intrinsics, get_camera_matrix_from_intrinsics

from tools.transformation import form_transf, transform_image_lightglue, estimate_pose_mine

import sys


class LightGlueOdometryReal():
    def __init__(self, image_width=None, image_height=None, datastr='real-time', batch_size=1, worker_num=6):
        self.image_height = image_height
        self.image_width = image_width

        if self.image_width is None or self.image_width is None:
            self.image_resize = False
        else:
            self.image_resize = True

        # self.test_dir = test_dir
        self.sequence_name = datastr
        self.output_path = self.sequence_name

        current_datetime = datetime.now()

        datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

        self.output_name = f"{self.sequence_name}+{datetime_string}"

        self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
        self.K = get_camera_matrix_from_intrinsics(self.focalx, self.focaly, self.centerx, self.centery)
        # self.P = get_extrinsic_matrix(self.sequence_name)

        # self.dataset = TrajFolderDatasetBetter(self.test_dir, rgb=False, transform=None, focalx=self.focalx,
        #                                        focaly=self.focaly,
        #                                        centerx=self.centerx, centery=self.centery)

        # self.testDataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=worker_num)
        # self.testDataiter = iter(self.testDataloader)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.set_grad_enabled(False)

        self.extractor = SuperPoint(max_num_keypoints=None).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        # self.matcher = torch.compile(self.matcher)

        self.output_directory = Path("results") / self.output_name
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.output_path = "./results/" + self.output_name

        self.image_current, self.image_next = None, None
        self.transformation_matrix = np.eye(4, dtype=np.float64)

        self.trajectory = []
        self.timestampContainer = list()

    def evaluate(self, image):
        if self.image_next is None:
            return self.evaluate_first(image)
        else:
            return self.evaluate_next(image)

    def publish_results(self):
        with open(self.output_path + "/results.txt", 'a') as file:
            file.write(str(self.timestampContainer[-1]) + ",")

            np.savetxt(file, self.trajectory[-1].reshape(-1, 12), delimiter=',', fmt='%.4f')

        result = self.trajectory[-1]

        odom = Odometry()

        x = result[0, 3]
        y = result[1, 3]
        z = result[2, 3]

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z

        rotation_mat = result[:3, :3]

        sy = math.sqrt(rotation_mat[0, 0] * rotation_mat[0, 0] + rotation_mat[1, 0] * rotation_mat[1, 0])
        singular = sy < 1e-6

        roll = None
        pitch = None
        yaw = None

        if not singular:
            roll = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
            pitch = math.atan2(-rotation_mat[2, 0], sy)
            yaw = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            roll = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            pitch = math.atan2(-rotation_mat[2, 0], sy)
            yaw = 0

        odom_quat = Quaternion(
            x=roll,
            y=pitch,
            z=yaw,
            w=yaw
        )
        odom.pose.pose.orientation = odom_quat
        # self.publisher.publish(odom)
        # self.get_logger().info(f'Published Odometry: Position ({x}, {y}, {z}), Orientation (Roll: {roll}, Pitch: {pitch}, Yaw: {yaw})')

        return odom


    def evaluate_first(self, image):
        self.image_next = image
        self.trajectory.append(self.transformation_matrix[:3, :])

        self.timestampContainer.append(float(datetime.now().timestamp()))

        return self.publish_results()

    def evaluate_next(self, image):
        self.image_current = self.image_next
        self.image_next = image

        image_current, scales0 = transform_image_lightglue(self.image_current)
        image_next, scales1 = transform_image_lightglue(self.image_next)

        image_current, image_next = image_current.to(self.device), image_next.to(self.device)

        assert scales0 == (1.0, 1.0)
        assert scales1 == (1.0, 1.0)

        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, image_current, image_next, self.device)

        kpts0_temp = feats0["keypoints"]
        kpts1_temp = feats1["keypoints"]
        matches = matches01["matches"]

        kpts0 = kpts0_temp[matches[:, 0]]
        kpts1 = kpts1_temp[matches[:, 1]]

        kpts0 = kpts0.detach().cpu().numpy()
        kpts1 = kpts1.detach().cpu().numpy()

        # K = scale_intrinsics(self.K, scales0)

        tresh = 1
        R, t = estimate_pose_mine(kpts0, kpts1, self.K, tresh)

        motion = form_transf(R, t)

        transformation_matrix = self.transformation_matrix @ np.linalg.inv(motion)
        self.trajectory.append(transformation_matrix[:3, :])

        self.timestampContainer.append(float(datetime.now().timestamp()))

        return self.publish_results()
