import numpy as np
import pandas as pd

from Tartanvo.evaluator.evaluate_rpe import ominus
from Tartanvo.evaluator.transformation import pos_quats2SEs, SEs2pos_quats, pos_quats2SEs2
from tools import ned2cam_mine


def pos_quat_timestamp_to_se3(input_path, output_path):
    df = pd.read_csv(input_path, sep=" ", header=None, comment="#")
    df = df.to_numpy()
    timestamp = df[:, 0]
    result = pos_quats2SEs(df[:, 1:])

    result_with_timestamp = np.column_stack((timestamp, result))

    np.savetxt(output_path, result_with_timestamp, delimiter=' ')


def shift0_mine_final(trajectory):
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


def trajectory_transform_mine_final(gt_trajectory, est_trajectory):
   gt_traj_trans = shift0_mine_final(gt_trajectory)
   est_traj_trans = shift0_mine_final(est_trajectory)

   # gt_traj_trans = to_cam_frame(gt_traj_trans)

   # gt_traj_trans = ned2cam_mine(gt_traj_trans)
   # est_traj_trans = ned2cam_mine(est_traj_trans)

   return gt_traj_trans, est_traj_trans


def rescale_mine_final(poses_gt, poses):
    '''
    :param poses_gt: se3
    :param poses: se3
    :return: rescaled poses
    '''
    assert poses_gt.shape == poses.shape
    length = poses_gt.shape[0]

    poses_scaled = np.zeros_like(poses)
    poses_scaled[0, :, :] = poses[0, :, :]

    scale_list = []

    for i in range(1, length):
        movement_gt = ominus(poses_gt[i - 1], poses_gt[i])
        movement = ominus(poses[i - 1], poses[i])

        translation_scale = np.linalg.norm(movement_gt[:3, 3], axis=0)
        translation_vector = movement[:3, 3]

        scale = translation_scale / np.max([np.linalg.norm(movement[:3, 3]), np.finfo(np.float64).eps])

        translation_vector_scaled = translation_vector * scale

        scale_list.append(scale)

        movement_scaled = np.eye(4)
        movement_scaled[:3, :3] = movement[:3, :3]
        movement_scaled[:3, 3] = translation_vector_scaled

        poses_scaled[i, :, :] = poses_scaled[i - 1, :, :] @ movement_scaled

    mean_scale = np.median(np.array(scale_list))
    return poses_scaled, mean_scale


def transform_trajectories_mine_final(gt_traj, est_traj, cal_scale=False):
   gt_traj, est_traj = trajectory_transform_mine_final(gt_traj, est_traj)
   if cal_scale :
      # gt_traj_quat = SEs2pos_quats(gt_traj)
      # est_traj_quat = SEs2pos_quats(est_traj)
      est_traj, s = rescale_mine_final(gt_traj, est_traj)
      print('  Scale, {}'.format(s))
      # est_traj = pos_quats2SEs2(est_traj_quat)
   else:
      s = 1.0
   return gt_traj, est_traj, s


if __name__ == "__main__":
    pos_quat_timestamp_to_se3(
        input_path="/home/washindeiru/studia/7_semestr/vo/papers/DEVOv2/DEVO/results/fpv_evs/2024-12-13_washindeiru_v3/Indoor_Forward_9_Davis_With_Gt_trial_0_step_DEVO/stamped_traj_estimate.txt",
        output_path="/home/washindeiru/studia/7_semestr/vo/papers/DEVOv2/DEVO/results/fpv_evs/2024-12-13_washindeiru_v3/Indoor_Forward_9_Davis_With_Gt_trial_0_step_DEVO/evaluate_mine/result_se3.txt"
    )
