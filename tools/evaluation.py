import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation

from Tartanvo.Datasets.transformation import SE2pos_quat
from SuperGlue.models.utils import compute_pose_error, angle_error_vec, angle_error_mat, compute_pose_error_mine
from Tartanvo.evaluator.evaluate_rpe import ominus
from Tartanvo.evaluator.evaluator_base import ATEEvaluator
from .transformation import transform_trajectories_mine


def form_transf(R, t):
   T = np.eye(4, dtype=np.float64)
   T[:3, :3] = R
   T[:3, 3] = t
   return T


def plot_path(sequence_name, ground_truth_path, predicted_path):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

   # gt_numpy = np.array(ground_truth_path)
   # pt_numpy = np.array(predicted_path)

   ax1.plot(ground_truth_path[:, 0], ground_truth_path[:, 1], label="ground truth")
   ax1.plot(predicted_path[:, 0], predicted_path[:, 1], label="prediction")
   ## to delete
   # ax1.set_xlim((-150, 50))
   ax1.set_xlabel("X")
   ax1.set_ylabel("Y")
   ax1.set_title("Trajectory")

   error = np.linalg.norm(ground_truth_path - predicted_path, axis=1)
   ax2.plot(np.arange(0, error.shape[0], 1), error, label="error")
   ax2.set_title("Error")
   ax2.set_xlabel("Frame")
   ax2.set_ylabel("Error")

   ax1.legend()
   # ax2.legend()

   try:
      os.mkdir(f"./results/{sequence_name}")
   except FileExistsError:
      pass

   fig.savefig(f"./results/{sequence_name}/results.png")

   fig.show()


def make_matrix_homogenous(matrix):
   temp = np.eye(4, dtype=np.float64)
   temp[:3, :3] = matrix[:3, :3]
   temp[:3, 3] = matrix[:3, 3]
   return temp



def plot_path_with_matrix(sequence_name, ground_truth_path_matrix, predicted_path_matrix):
   ground_truth_path = ground_truth_path_matrix[:, [0, 1], [3, 3]]
   predicted_path = predicted_path_matrix[:, [0, 1], [3, 3]]
   np.savetxt(f"./results/{sequence_name}/results.txt", predicted_path_matrix.reshape(-1, 12), delimiter=' ')
   plot_path(sequence_name, ground_truth_path, predicted_path)


def plot_path_with_matrix_and_angle(sequence_name, ground_truth_path_matrix, predicted_path_matrix, error_angle_list):
   ground_truth_path = ground_truth_path_matrix[:, [0, 1], [3, 3]]
   predicted_path = predicted_path_matrix[:, [0, 1], [3, 3]]

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
   ax1.plot(ground_truth_path[:, 0], ground_truth_path[:, 1], label="ground truth")
   ax1.plot(predicted_path[:, 0], predicted_path[:, 1], label="prediction")
   ax1.set_xlabel("X")
   ax1.set_ylabel("Y")
   ax1.set_title("Trajectory")

   error_angle_matrix = np.array(error_angle_list, dtype=np.float64)
   ax2.plot(np.arange(0, error_angle_matrix.shape[0], 1), error_angle_matrix[:, 0])
   ax2.set_title("Rotation error angle")
   ax2.set_xlabel("Frame")
   ax2.set_ylabel("Angle error (degrees)")

   ax3.plot(np.arange(0, error_angle_matrix.shape[0], 1), error_angle_matrix[:, 1])
   ax3.set_title("Translation error angle")
   ax3.set_xlabel("Frame")
   ax3.set_ylabel("Angle error (degrees)")

   ax1.legend()

   os.makedirs(f"./results/{sequence_name}", exist_ok=True)
   fig.savefig(f"./results/{sequence_name}/results_angle.png")
   fig.show()


def save_as_quat(sequence_name, predicted_path_matrix):
   quat_matrix = np.zeros((predicted_path_matrix.shape[0], 7), dtype=np.float64)
   for i in range(0, predicted_path_matrix.shape[0]):
      quat_matrix[i, :] = SE2pos_quat(predicted_path_matrix[i, :, :])

   np.savetxt(f"./results/{sequence_name}/results_quat.txt", quat_matrix, delimiter=' ')


def save_as_s3(sequence_name, predicted_path_matrix):
   np.savetxt(f"./results/{sequence_name}/results_s3.txt", predicted_path_matrix.reshape(-1, 12), delimiter=' ')


# def save_as_s3_with_timestamp(sequence_name, predicted_path_matrix, timestamps):
#    path_matrix = predicted_path_matrix.reshape(-1, 12)
#
#    df = pd.DataFrame({
#       "timestamp": timestamps,
#       "a1": path_matrix[:, 0],
#       "a2": path_matrix[:, 1],
#       "a3": path_matrix[:, 2],
#       "a4": path_matrix[:, 3],
#       "b1": path_matrix[:, 4],
#       "b2": path_matrix[:, 5],
#       "b3": path_matrix[:, 6],
#       "b4": path_matrix[:, 7],
#       "c1": path_matrix[:, 8],
#       "c2": path_matrix[:, 9],
#       "c3": path_matrix[:, 10],
#       "c4": path_matrix[:, 11]
#    })
#
#    df.to_csv(f"./results/{sequence_name}/results_s3_with_timestamp.txt", sep=" ", index=False, header=False)


def save_as_s3_with_timestamp_experimental(sequence_name, predicted_path_matrix, timestamps):
   timestamp_matrix = np.array(timestamps).reshape(-1, 1)
   path_matrix = predicted_path_matrix.reshape(-1, 12)

   result = np.concatenate((timestamp_matrix, path_matrix), axis=1)

   np.savetxt(f"./results/{sequence_name}/results_s3_with_timestamp.txt", result, delimiter=' ')


def save_3d_plot(sequence_name, predicted_path_matrix):
   predicted_path_matrix = predicted_path_matrix.reshape(-1, 12)
   fig = plt.figure(figsize = (7, 6))
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(predicted_path_matrix[:, 3], predicted_path_matrix[:, 7], predicted_path_matrix[:, 11])
   ax.set_xlabel('x (m)')
   ax.set_ylabel('y (m)')
   ax.set_zlabel('z (m)')

   os.makedirs(f"./results/{sequence_name}", exist_ok=True)
   fig.savefig(f"./results/{sequence_name}/results_3d.png")


def compute_pose_error_better(previous_ground_truth, current_ground_truth, rotation, translation):
   transformation_ground_truth = np.linalg.inv(make_matrix_homogenous(previous_ground_truth)) @ make_matrix_homogenous(current_ground_truth)
   return compute_pose_error(transformation_ground_truth, rotation, translation)


def quaternion_to_angles(quaternion):
   angles = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
   return angles


def evaluate_trajectory_from_tartan(traj_gt, traj_est):
   assert traj_gt.shape[0] == traj_est.shape[0]

   error_result = []

   for i in range(traj_gt.shape[0] - 1):
      # error44 = ominus( ominus(traj_est[i], traj_est[i + 1]),
      #                   ominus(traj_gt[i], traj_gt[i + 1]) )
      #
      # trans = angle_error_vec()
      est_trans = ominus(traj_est[i], traj_est[i + 1])
      gt_trans = ominus(traj_gt[i], traj_gt[i + 1])

      r_gt = gt_trans[:3, :3]
      t_gt = gt_trans[:3, 3]
      r_est = est_trans[:3, :3]
      t_est = est_trans[:3, 3]

      translation_error, rotation_error = compute_pose_error_mine(r_gt, t_gt, r_est, t_est)
      #
      # translation_error = angle_error_vec(est_trans[:3, 3], gt_trans[:3, 3])
      # rotation_error = angle_error_mat(est_trans[:3, :3], gt_trans[:3, :3])

      error_result.append([translation_error, rotation_error])

   return error_result


def evaluate(gt_traj, est_traj):
   result = evaluate_trajectory_from_tartan(gt_traj, est_traj)
   result = np.array(result)

   translation_error = result[:, 0]
   rotation_error = result[:, 1]

   translation_error_mean = np.mean(translation_error)
   rotation_error_mean = np.mean(rotation_error)

   translation_error_median = np.median(translation_error)
   rotation_error_median = np.median(rotation_error)

   return translation_error, rotation_error, translation_error_mean, rotation_error_mean, translation_error_median, rotation_error_median


def evaluate_one_trajectory(gt_traj, est_traj):
   gt_traj_trans, est_traj_trans, s = transform_trajectories_mine(gt_traj, est_traj)

   ate_eval = ATEEvaluator()
   ate_score, gt_ate_aligned, est_ate_aligned = ate_eval.evaluate(gt_traj, est_traj, False)
   translation_error, rotation_error, translation_error_mean, rotation_error_mean = evaluate(gt_traj_trans, est_traj_trans)

   return {
      'ate_score': ate_score,
      'gt_aligned': gt_ate_aligned,
      'est_aligned': est_ate_aligned,
      'translation_error': translation_error,
      'rotation_error': rotation_error,
      'translation_error_mean': translation_error_mean,
      'rotation_error_mean': rotation_error_mean
   }




