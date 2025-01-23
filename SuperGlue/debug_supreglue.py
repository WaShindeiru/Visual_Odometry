import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import torch
import cv2
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import make_matching_plot, scale_intrinsics, estimate_pose
from Tartanvo.Datasets.utils import dataset_intrinsics, get_camera_matrix_from_intrinsics
from tools import transform_image, form_transf, get_extrinsic_matrix, get_pose_stolen


def test_kitti():
   matplotlib.use('TkAgg')

   config = {
      'superpoint': {
         'nms_radius': 4,
         'keypoint_threshold': 0.01,
         'max_keypoints': 1024
      },
      'superglue': {
         # 'indoor', 'outdoor'
         'weights': 'outdoor',
         'sinkhorn_iterations': 20,
         'match_threshold': 0.5,
      }
   }

   # config = {
   #    'superpoint': {
   #       'nms_radius': 4,
   #       'keypoint_threshold': 0.04,
   #       'max_keypoints': 500
   #    },
   #    'superglue': {
   #       # 'indoor', 'outdoor'
   #       'weights': 'outdoor',
   #       'sinkhorn_iterations': 20,
   #       'match_threshold': 0.5,
   #    }
   # }

   matching = Matching(config).eval().to('cuda')

   datastr = "kitti"
   path = "/media/washindeiru/Hard Disc/odom_files/kitti/02/image_0"

   # focalx, focaly, centerx, centery = dataset_intrinsics(datastr)
   # K_old = get_camera_matrix_from_intrinsics(focalx, focaly, centerx, centery)
   # P = get_extrinsic_matrix(datastr)

   image1 = cv2.imread(path + "/000057.png", 0)
   image2 = cv2.imread(path + "/000058.png", 0)

   width = 630
   height = 470

   image0, resized_first_image, scales0 = transform_image(image1, 'cuda', (width, height))
   image1, resized_second_image, scales1 = transform_image(image2, 'cuda', (width, height))

   result = matching({'image0': resized_first_image, 'image1': resized_second_image})
   result = {key: value[0].cpu().detach().numpy() for key, value in result.items()}

   kpts0, kpts1 = result['keypoints0'], result['keypoints1']
   matches, conf = result['matches0'], result['matching_scores0']

   valid = matches > -1
   mkpts0 = kpts0[valid]
   mkpts1 = kpts1[matches[valid]]
   mconf = conf[valid]

   text = [
      'SuperGlue',
      'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
      'Matches: {}'.format(len(mkpts0)),
   ]

   # color = cm.jet(mconf)
   color = np.concatenate([np.zeros((mconf.shape[0], 1)) * 0.5, np.ones((mconf.shape[0], 1)), np.zeros((mconf.shape[0], 1)), np.ones((mconf.shape[0], 1))], axis=1)

   # jet_data = cm.jet(np.linspace(0, 1, 256))
   #
   # # Modify the colormap data to include green at a specific position
   # # Example: Insert green at the midpoint (position 0.5)
   # green_color = np.array([0, 1, 0, 1])  # RGBA for green
   # midpoint = len(jet_data) // 2
   # jet_data[midpoint] = green_color
   #
   # # Create a new colormap using the modified data
   # custom_jet = LinearSegmentedColormap.from_list('custom_jet', jet_data)

   make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
                      '/home/washindeiru/studia/7_semestr/vo/visual_odometry/SuperGlue/results/' + datastr + str(datetime.now().timestamp()) + '.png',
                      show_keypoints=True,
                      fast_viz=True)

   # K = scale_intrinsics(K_old, scales0)
   #
   # thresh = 1.  # In pixels relative to resized image size.
   # ret = estimate_pose(mkpts0, mkpts1, K, K, thresh)
   #
   # # motion = get_pose_stolen(K, P, mkpts0, mkpts1)
   #
   # if ret is None:
   #    R, t = np.inf, np.inf
   # else:
   #    R, t, inliers = ret
   #
   # motion = form_transf(R, t)
   #
   # print(motion)


def test_m2dgr():
   matplotlib.use('TkAgg')

   config = {
      'superpoint': {
         'nms_radius': 4,
         'keypoint_threshold': 0.1,
         'max_keypoints': 1024
      },
      'superglue': {
         # 'indoor', 'outdoor'
         'weights': 'outdoor',
         'sinkhorn_iterations': 20,
         'match_threshold': 0.5,
      }
   }

   # config = {
   #    'superpoint': {
   #       'nms_radius': 4,
   #       'keypoint_threshold': 0.005,
   #       'max_keypoints': 1024
   #    },
   #    'superglue': {
   #       # 'indoor', 'outdoor'
   #       'weights': 'outdoor',
   #       'sinkhorn_iterations': 20,
   #       'match_threshold': 0.2,
   #    }
   # }

   matching = Matching(config).eval().to('cuda')

   datastr = "m2dgr_calib"
   path = "/media/washindeiru/Hard Disc/odom_files/kitti/02/image_0"

   focalx, focaly, centerx, centery = dataset_intrinsics(datastr)
   K_old = get_camera_matrix_from_intrinsics(focalx, focaly, centerx, centery)
   # P = get_extrinsic_matrix(datastr)

   image1 = cv2.imread(path + "/1628058877436989784.png", 0)
   image2 = cv2.imread(path + "/1628058877503411055.png", 0)

   width = 630
   height = int(width // 1.3432)

   image0, resized_first_image, scales0 = transform_image(image1, 'cuda', (width, height))
   image1, resized_second_image, scales1 = transform_image(image2, 'cuda', (width, height))

   result = matching({'image0': resized_first_image, 'image1': resized_second_image})
   result = {key: value[0].cpu().detach().numpy() for key, value in result.items()}

   kpts0, kpts1 = result['keypoints0'], result['keypoints1']
   matches, conf = result['matches0'], result['matching_scores0']

   valid = matches > -1
   mkpts0 = kpts0[valid]
   mkpts1 = kpts1[matches[valid]]
   mconf = conf[valid]

   text = [
      'SuperGlue',
      'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
      'Matches: {}'.format(len(mkpts0)),
   ]

   color = cm.jet(mconf)

   make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
                      '/home/washindeiru/studia/7_semestr/vo/visual_odometry/SuperGlue/results/m2dgr_resize.png',
                      show_keypoints=True,
                      fast_viz=True)

   K = scale_intrinsics(K_old, scales0)

   thresh = 1.  # In pixels relative to resized image size.
   ret = estimate_pose(mkpts0, mkpts1, K, K, thresh)

   # motion = get_pose_stolen(K, P, mkpts0, mkpts1)

   if ret is None:
      R, t = np.inf, np.inf
   else:
      R, t, inliers = ret

   motion = form_transf(R, t)

   print(motion)



if __name__ == "__main__":
   test_kitti()