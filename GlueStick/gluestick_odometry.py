from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from gluestick import numpy_image_to_torch, batch_to_np
from GlueStick.gluestick.models.two_view_pipeline import TwoViewPipeline
from Tartanvo.Datasets.tartanTrajFlowDatasetBetter import TrajFolderDatasetBetter
from Tartanvo.Datasets.utils import dataset_intrinsics, get_camera_matrix_from_intrinsics
from tools import save_as_s3_with_timestamp_experimental

from tools.transformation import estimate_pose_mine, form_transf

class GlueStickOdometry:
    def __init__(self, image_width=None, image_height=None, datastr='davis_9_calib',
                 test_dir="/media/washindeiru/New Volume/backup/windowsBackup16-01-2025/g/odom_files/DEVOv2/DEVO/data/indoor_forward_3_davis_with_gt/img_short",
                 batch_size=1, worker_num=6):

        self.image_height = image_height
        self.image_width = image_width

        if self.image_width is None or self.image_width is None:
            self.image_resize = False
        else:
            self.image_resize = True

        self.test_dir = test_dir
        self.sequence_name = datastr
        self.output_path = self.sequence_name

        self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
        self.K = get_camera_matrix_from_intrinsics(self.focalx, self.focaly, self.centerx, self.centery)
        # self.P = get_extrinsic_matrix(self.sequence_name)

        self.dataset = TrajFolderDatasetBetter(self.test_dir, rgb=False, transform=None, focalx=self.focalx,
                                               focaly=self.focaly,
                                               centerx=self.centerx, centery=self.centery)

        self.testDataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=worker_num)
        self.testDataiter = iter(self.testDataloader)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.set_grad_enabled(False)

        MAX_N_POINTS, MAX_N_LINES = 1000, 300

        # Evaluation config
        conf = {
            'name': 'two_view_pipeline',
            'use_lines': False,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {
                    'force_num_keypoints': False,
                    'max_num_keypoints': MAX_N_POINTS,
                },
                'wireframe_params': {
                    'merge_points': True,
                    'merge_line_endpoints': True,
                },
                'max_n_lines': 0,
            },
            'matcher': {
                'name': 'gluestick',
                'weights': "/home/washindeiru/studia/7_semestr/vo/visual_odometry/GlueStick/gluestick/weights/checkpoint_GlueStick_MD.tar",
                'trainable': False,
            },
            'ground_truth': {
                'from_pose_depth': False,
            }
        }

        self.pipeline = TwoViewPipeline(conf).eval().to(self.device)
        # self.pipeline = torch.compile(self.pipeline)


        self.output_directory = Path("results") / self.output_path
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

            torch_gray0, torch_gray1 = numpy_image_to_torch(image_current), numpy_image_to_torch(image_next)
            torch_gray0, torch_gray1 = torch_gray0.to(self.device)[None], torch_gray1.to(self.device)[None]
            x = {'image0': torch_gray0, 'image1': torch_gray1}
            prediction = self.pipeline(x)

            pred = batch_to_np(prediction)
            kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
            m0 = pred["matches0"]

            valid_matches = m0 != -1
            match_indices = m0[valid_matches]
            matched_kps0 = kp0[valid_matches]
            matched_kps1 = kp1[match_indices]

            tresh = 1.
            R, t = estimate_pose_mine(matched_kps0, matched_kps1, self.K, tresh)

            motion = form_transf(R, t)

            transformation_matrix = transformation_matrix @ np.linalg.inv(motion)
            trajectory[i + 1, :, :] = transformation_matrix[:3, :]

            timestampContainer.append(sample['timestamp'][0])

        save_as_s3_with_timestamp_experimental(self.output_path, trajectory, timestampContainer)


if __name__ == "__main__":
    odom = GlueStickOdometry()
    odom.visual_odometry(subset = None)
