from typing import Union, List, Optional

import torch

import numpy as np
import cv2

from SuperGlue.models.utils import process_resize, frame2tensor
from Tartanvo.evaluator.trajectory_transform import rescale
from Tartanvo.evaluator.transformation import SE2pos_quat, pos_quat2SE, SEs2pos_quats, pos_quats2SEs2


def shift0_mine(trajectory):
   '''
   :param trajectory: sequence of SE(3)
   :return: translate and rotate the trajectory
   '''
   traj_init = trajectory[0]
   traj_init_inv = np.linalg.inv(traj_init)
   new_trajectory = list()
   for tt in trajectory:
      ttt = traj_init_inv.dot(tt)
      new_trajectory.append(ttt)
   return np.array(new_trajectory)


def trajectory_transform_mine(gt_trajectory, est_trajectory):
   gt_traj_trans = shift0_mine(gt_trajectory)
   est_traj_trans = shift0_mine(est_trajectory)

   # gt_traj_trans = to_cam_frame(gt_traj_trans)

   # gt_traj_trans = ned2cam_mine(gt_traj_trans)
   # est_traj_trans = ned2cam_mine(est_traj_trans)

   return gt_traj_trans, est_traj_trans


def ned2cam_mine(traj):
   # T = np.array([ [0., 0., 1., 0],
   #      [-1, 0., 0., 0],
   #      [0., -1, 0., 0],
   #      [0., 0., 0., 1. ]] , dtype=np.float32)

   # T = np.array([[0, 1, 0, 0],
   #               [0, 0, 1, 0],
   #               [1, 0, 0, 0],
   #               [0, 0, 0, 1]], dtype=np.float32)

   T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

   T_inv = np.linalg.inv(T)
   new_traj = []

   for tt in traj:
      ttt= T_inv @ tt @ T
      new_traj.append(ttt)

   return np.array(new_traj)


def transform_trajectories_mine(gt_traj, est_traj, cal_scale=False):
   gt_traj, est_traj = trajectory_transform_mine(gt_traj, est_traj)
   if cal_scale :
      gt_traj_quat = SEs2pos_quats(gt_traj)
      est_traj_quat = SEs2pos_quats(est_traj)
      est_traj_quat, s = rescale(gt_traj_quat, est_traj_quat)
      print('  Scale, {}'.format(s))
      est_traj = pos_quats2SEs2(est_traj_quat)
   else:
      s = 1.0
   return gt_traj, est_traj, s


# supreglue
def transform_image(image, device, resize=[-1], resize_float=None, rotation=0):
   if image is None:
      return None, None, None
   w, h = image.shape[1], image.shape[0]
   w_new, h_new = process_resize(w, h, resize)
   scales = (float(w) / float(w_new), float(h) / float(h_new))

   if resize_float:
      image = cv2.resize(image.astype('float32'), (w_new, h_new))
   else:
      image = cv2.resize(image, (w_new, h_new)).astype('float32')

   # if rotation != 0:
   #    image = np.rot90(image, k=rotation)
   #    if rotation % 2:
   #       scales = scales[::-1]

   inp = frame2tensor(image, device)
   return image, inp, scales


def decompose_essential_mat(E, q1, q2, intrinsic_matrix, extrinsic_matrix):
   """
   Decompose the Essential matrix

   Parameters
   ----------
   E (ndarray): Essential matrix
   q1 (ndarray): The good keypoints matches position in i-1'th image
   q2 (ndarray): The good keypoints matches position in i'th image

   Returns
   -------
   right_pair (list): Contains the rotation matrix and translation vector
   """

   R1, R2, t = cv2.decomposeEssentialMat(E)
   T1 = form_transf(R1, np.ndarray.flatten(t))
   T2 = form_transf(R2, np.ndarray.flatten(t))
   T3 = form_transf(R1, np.ndarray.flatten(-t))
   T4 = form_transf(R2, np.ndarray.flatten(-t))
   transformations = [T1, T2, T3, T4]

   # Homogenize K
   K = np.concatenate((intrinsic_matrix, np.zeros((3, 1))), axis=1)
   # print(f"Before homogenization: {self.K}")
   # print(f"After homogenization: {K}")

   # List of projections
   projections = [K @ T1, K @ T2, K @ T3, K @ T4]

   np.set_printoptions(suppress=True)

   # print ("\nTransform 1\n" +  str(T1))
   # print ("\nTransform 2\n" +  str(T2))
   # print ("\nTransform 3\n" +  str(T3))
   # print ("\nTransform 4\n" +  str(T4))

   positives = []
   for P, T in zip(projections, transformations):
      hom_Q1 = cv2.triangulatePoints(extrinsic_matrix, P, q1.T, q2.T)
      hom_Q2 = T @ hom_Q1
      # Un-homogenize
      Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
      Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

      total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
      relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                               np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
      positives.append(total_sum + relative_scale)

   # Decompose the Essential matrix using built in OpenCV function
   # Form the 4 possible transformation matrix T from R1, R2, and t
   # Create projection matrix using each T, and triangulate points hom_Q1
   # Transform hom_Q1 to second camera using T to create hom_Q2
   # Count how many points in hom_Q1 and hom_Q2 with positive z value
   # Return R and t pair which resulted in the most points with positive z

   max = np.argmax(positives)
   if (max == 2):
      # print(-t)
      return R1, np.ndarray.flatten(-t)
   elif (max == 3):
      # print(-t)
      return R2, np.ndarray.flatten(-t)
   elif (max == 0):
      # print(t)
      return R1, np.ndarray.flatten(t)
   elif (max == 1):
      # print(t)
      return R2, np.ndarray.flatten(t)


# def decompose_essential_mat(E, intrinsic_matrix, q1, q2):
#    """
#    Decompose the Essential matrix
#
#    Parameters
#    ----------
#    E (ndarray): Essential matrix
#    q1 (ndarray): The good keypoints matches position in i-1'th image
#    q2 (ndarray): The good keypoints matches position in i'th image
#
#    Returns
#    -------
#    right_pair (list): Contains the rotation matrix and translation vector
#    """
#
#    R1, R2, t = cv2.decomposeEssentialMat(E)
#    T1 = form_transf(R1, np.ndarray.flatten(t))
#    T2 = form_transf(R2, np.ndarray.flatten(t))
#    T3 = form_transf(R1, np.ndarray.flatten(-t))
#    T4 = form_transf(R2, np.ndarray.flatten(-t))
#    transformations = [T1, T2, T3, T4]
#
#    # Homogenize K
#    K = np.concatenate((intrinsic_matrix, np.zeros((3, 1))), axis=1)
#    # print(f"Before homogenization: {self.K}")
#    # print(f"After homogenization: {K}")
#
#    # List of projections
#    projections = [K @ T1, K @ T2, K @ T3, K @ T4]
#
#    np.set_printoptions(suppress=True)
#
#    # print ("\nTransform 1\n" +  str(T1))
#    # print ("\nTransform 2\n" +  str(T2))
#    # print ("\nTransform 3\n" +  str(T3))
#    # print ("\nTransform 4\n" +  str(T4))
#
#    positives = []
#    for P, T in zip(projections, transformations):
#       hom_Q1 = cv2.triangulatePoints(self.dataset_handler.projectionMatrix_left, P, q1.T, q2.T)
#       hom_Q2 = T @ hom_Q1
#       # Un-homogenize
#       Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
#       Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
#
#       total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
#       relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
#                                np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
#       positives.append(total_sum + relative_scale)
#
#    # Decompose the Essential matrix using built in OpenCV function
#    # Form the 4 possible transformation matrix T from R1, R2, and t
#    # Create projection matrix using each T, and triangulate points hom_Q1
#    # Transform hom_Q1 to second camera using T to create hom_Q2
#    # Count how many points in hom_Q1 and hom_Q2 with positive z value
#    # Return R and t pair which resulted in the most points with positive z
#
#    max = np.argmax(positives)
#    if (max == 2):
#       # print(-t)
#       return R1, np.ndarray.flatten(-t)
#    elif (max == 3):
#       # print(-t)
#       return R2, np.ndarray.flatten(-t)
#    elif (max == 0):
#       # print(t)
#       return R1, np.ndarray.flatten(t)
#    elif (max == 1):
#       # print(t)
#       return R2, np.ndarray.flatten(t)
#
#
def form_transf(R, t):
   T = np.eye(4, dtype=np.float64)
   T[:3, :3] = R
   T[:3, 3] = t
   return T


def get_pose_stolen(intrinsic_matrix, extrinsic_matrix, q1, q2):
   """
   Calculates the transformation matrix

   Parameters
   ----------
   q1 (ndarray): The good keypoints matches position in i-1'th image
   q2 (ndarray): The good keypoints matches position in i'th image

   Returns
   -------
   transformation_matrix (ndarray): The transformation matrix
   """

   essential, mask = cv2.findEssentialMat(q1, q2, intrinsic_matrix)
   # print("\nEssential matrix:\n" + str(Essential))

   R, t = decompose_essential_mat(essential, q1, q2, intrinsic_matrix, extrinsic_matrix)

   return form_transf(R, t)


def get_extrinsic_matrix(name):
   data = None
   if name == "m2dgr" or name == "m2dgr_calib":
      data = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.]])
      # data = np.array([[ 0., 0., 1., 0.30456],
      #                   [-1, 0., 0., 0.00065],
      #                   [0., -1, 0., 0.65376]])
   else:
      raise Exception(f"wrong name: {name}")
   return data


def resize_image_lightglue(
        image: np.ndarray,
        size: Union[List[int], int],
        fn: str = "max",
        interp: Optional[str] = "area",
):
   """Resize an image to a fixed size, or according to max or min edge."""
   h, w = image.shape[:2]

   fn = {"max": max, "min": min}[fn]
   if isinstance(size, int):
      scale = size / fn(h, w)
      h_new, w_new = int(round(h * scale)), int(round(w * scale))
      scale = (w_new / w, h_new / h)
   elif isinstance(size, (tuple, list)):
      h_new, w_new = size
      scale = (w_new / w, h_new / h)
   else:
      raise ValueError(f"Incorrect new size: {size}")
   mode = {
      "linear": cv2.INTER_LINEAR,
      "cubic": cv2.INTER_CUBIC,
      "nearest": cv2.INTER_NEAREST,
      "area": cv2.INTER_AREA,
   }[interp]
   return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
   """Normalize the image tensor and reorder the dimensions."""
   if image.ndim == 3:
      image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
   elif image.ndim == 2:
      image = image[None]  # add channel axis
   else:
      raise ValueError(f"Not an image: {image.shape}")
   return torch.tensor(image / 255.0, dtype=torch.float)


def transform_image_lightglue(image, resize: int = None):
   scale = (1.0, 1.0)
   if resize is not None:
      image, scale = resize_image_lightglue(image, resize)
   return numpy_image_to_torch(image), scale


def estimate_pose_mine(kpts0, kpts1, K, thresh, conf=0.99999):

   E, mask = cv2.findEssentialMat(kpts0, kpts1, K, threshold=thresh, prob=conf, method=cv2.RANSAC)

   assert E is not None

   # result = cv2.recoverPose(E=E, points1=kpts0, points2=kpts1, distCoeffs1=np.array([-0.436107594318055,0.166413618922992,0,0]),
   #                     distCoeffs2=np.array([-0.436107594318055, 0.166413618922992, 0, 0]),
   #                                   cameraMatrix1=K, cameraMatrix2=K, mask=mask)
   #
   # i, _, R, t, _ = result

   result = cv2.recoverPose(E=E, points1=kpts0, points2=kpts1, cameraMatrix=K, mask=mask)

   a, R, t, i = result

   return R, t[:, 0]