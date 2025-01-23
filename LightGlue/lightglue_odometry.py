from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LightGlue.lightglue import SuperPoint, LightGlue, match_pair
from Tartanvo.Datasets.tartanTrajFlowDatasetBetter import TrajFolderDatasetBetter
from Tartanvo.Datasets.utils import dataset_intrinsics, get_camera_matrix_from_intrinsics
from tools import transform_image_lightglue, save_as_s3_with_timestamp_experimental

from tools.transformation import transform_image, estimate_pose_mine, form_transf


class LightGlueOdometry():
    def __init__(self, image_width = None, image_height = None, datastr = "davis_3_calib",
                 test_dir="/media/washindeiru/New Volume/backup/windowsBackup16-01-2025/g/odom_files/DEVOv2/DEVO/data/indoor_forward_3_davis_with_gt/img_short",
                 batch_size = 1, worker_num = 6):

        self.image_height = image_height
        self.image_width = image_width

        if self.image_width is None or self.image_width is None:
            self.image_resize = False
        else:
            self.image_resize = True

        self.test_dir = test_dir
        self.sequence_name = datastr

        current_datetime = datetime.now()

        datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

        self.output_name = f"{self.sequence_name}+{datetime_string}"

        self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
        self.K = get_camera_matrix_from_intrinsics(self.focalx, self.focaly, self.centerx, self.centery)
        # self.P = get_extrinsic_matrix(self.sequence_name)

        self.dataset = TrajFolderDatasetBetter(self.test_dir, rgb=False, transform=None, focalx=self.focalx,
                                               focaly=self.focaly,
                                               centerx=self.centerx, centery=self.centery)

        self.testDataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=worker_num)
        self.testDataiter = iter(self.testDataloader)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        torch.set_grad_enabled(False)

        self.extractor = SuperPoint(max_num_keypoints=None).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        # self.matcher = torch.compile(self.matcher)

        self.output_directory = Path("results") / self.output_name
        self.output_directory.mkdir(parents=True, exist_ok=True)


    def visual_odometry(self, subset = None):
        if subset is None:
            num_frames = len(self.dataset) + 1
        else:
            num_frames = subset

        transformation_matrix = np.eye(4, dtype=np.float64)

        trajectory = np.zeros((num_frames, 3, 4))
        trajectory[0] = transformation_matrix[:3, :]

        timestampContainer = list()
        timestampContainer.append(float(self.dataset.getFirstTimestamp()))

        for i in tqdm(range(num_frames - 1)):
            sample = next(self.testDataiter)

            image_current = sample['img1'].numpy().squeeze()
            image_next = sample['img2'].numpy().squeeze()

            # image1, image_current, scales0 = transform_image(image_current, self.device)
            # image2, image_next, scales1 = transform_image(image_next, self.device)
            #
            # assert scales0 == (1.0, 1.0)
            # assert scales1 == (1.0, 1.0)

            image_current, scales0 = transform_image_lightglue(image_current)
            image_next, scales1 = transform_image_lightglue(image_next)

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

            transformation_matrix = transformation_matrix @ np.linalg.inv(motion)
            trajectory[i + 1, :, :] = transformation_matrix[:3, :]

            timestampContainer.append(float(sample['timestamp'][0]))

        save_as_s3_with_timestamp_experimental(self.output_name, trajectory, timestampContainer)


if __name__ == "__main__":
    vo = LightGlueOdometry()
    vo.visual_odometry(subset=None)