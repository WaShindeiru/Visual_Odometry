import torch
import numpy as np
from pathlib import Path

from datetime import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm

from Tartanvo.Datasets.tartanTrajFlowDatasetBetter import TrajFolderDatasetBetter
from Tartanvo.Datasets.utils import dataset_intrinsics, get_camera_matrix_from_intrinsics
from models.matching import Matching
from models.utils import *
from tools import *

# width = 630
# height = int(width // 1.3432)
# 480, 640

class SuperGlueOdometry_davis():
   def __init__(self, image_width = 630, image_height = int(630 // 1.3432), datastr = "davis_3_calib",
                test_dir="/media/washindeiru/840265A302659B46/odom_files/DEVOv2/DEVO/data/indoor_forward_3_davis_with_gt/img_short",
                batch_size = 1, worker_num = 6):

      self.image_height = image_height
      self.image_width = image_width

      self.test_dir = test_dir
      self.sequence_name = datastr

      current_datetime = datetime.now()

      # Format the date and time as a string
      datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

      self.output_name = f"{self.sequence_name}+{datetime_string}"

      self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
      self.K = get_camera_matrix_from_intrinsics(self.focalx, self.focaly, self.centerx, self.centery)
      # self.P = get_extrinsic_matrix(self.sequence_name)

      self.dataset = TrajFolderDatasetBetter(self.test_dir, rgb=False, transform=None, focalx=self.focalx, focaly=self.focaly,
                                                  centerx=self.centerx, centery=self.centery)

      self.testDataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=worker_num)
      self.testDataiter = iter(self.testDataloader)

      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      # assert self.device == 'cuda'

      config = {
         'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
         },
         'superglue': {
            # 'indoor', 'outdoor'
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
         }
      }

      # config = {
      #    'superpoint': {
      #       'nms_radius': 4,
      #       'keypoint_threshold': 0.1,
      #       'max_keypoints': 1024
      #    },
      #    'superglue': {
      #       # 'indoor', 'outdoor'
      #       'weights': 'outdoor',
      #       'sinkhorn_iterations': 20,
      #       'match_threshold': 0.5,
      #    }
      # }

      self.matching = Matching(config).eval().to(self.device)

      self.output_directory = Path("results") / self.output_name
      self.output_directory.mkdir(parents=True, exist_ok=True)

   def visual_odometry(self, subset: int = None):
      if subset is None:
         num_frames = len(self.dataset) + 1
      else:
         num_frames = subset

      transformation_matrix = np.eye(4, dtype=np.float64)

      trajectory = np.zeros((num_frames, 3, 4))
      trajectory[0] = transformation_matrix[:3, :]

      timestampContainer = list()
      timestampContainer.append(float(self.dataset.getFirstTimestamp()))

      invalid_count = 0

      for i in tqdm(range(num_frames - 1)):
         sample = next(self.testDataiter)

         image_current = sample['img1'].numpy().squeeze()
         image_next = sample['img2'].numpy().squeeze()

         image1, resized_first_image, scales0 = transform_image(image_current, 'cuda', (self.image_width, self.image_height))
         image2, resized_second_image, scales1 = transform_image(image_next, 'cuda', (self.image_width, self.image_height))
         # image0, resized_first_image, scales0 = transform_image(image_current, 'cuda')
         # image1, resized_second_image, scales1 = transform_image(image_next, 'cuda')

         result = self.matching({'image0': resized_first_image, 'image1': resized_second_image})
         result = {key: value[0].cpu().detach().numpy() for key, value in result.items()}

         kpts0, kpts1 = result['keypoints0'], result['keypoints1']
         matches, conf = result['matches0'], result['matching_scores0']

         valid = matches > -1
         mkpts0 = kpts0[valid]
         mkpts1 = kpts1[matches[valid]]
         mconf = conf[valid]

         K = scale_intrinsics(self.K, scales0)

         tresh = 1  # In pixels relative to resized image size.
         # ret = estimate_pose(mkpts0, mkpts1, K, K, tresh)
         R, t = estimate_pose_mine(mkpts0, mkpts1, K, tresh)

         # if ret is None:
         #    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
         #    t = np.array([0, 0, 0])
         #
         #    invalid_count += 1
         #
         # else:
         #    R, t, inliers = ret

         # R, t, inliers = ret

         motion = form_transf(R, t)

         # motion = get_pose_stolen(K, self.P, mkpts0, mkpts1)

         transformation_matrix = transformation_matrix @ np.linalg.inv(motion)
         trajectory[i+1, :, :] = transformation_matrix[:3, :]

         timestampContainer.append(float(sample['timestamp'][0]))

      save_3d_plot(self.output_name, trajectory)
      save_as_s3_with_timestamp_experimental(self.output_name, trajectory, timestampContainer)

      print(f"Invalid count: {invalid_count}")


if __name__ == "__main__":

   vo = SuperGlueOdometry_davis()
   vo.visual_odometry(subset=None)
