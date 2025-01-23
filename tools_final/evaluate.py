import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import make_matrix_homogenous, evaluate
from tools_final.tools_final_mine import transform_trajectories_mine_final


def make_dataset_homogenous(matrix):
   result = np.zeros((matrix.shape[0], 4, 4))

   for i in range(0, matrix.shape[0]):
      temp = matrix[i, :].reshape(3, 4)
      result[i, :, :] = make_matrix_homogenous(temp)

   return result


def separate_timestamps(traj):
   est_traj = traj[:, 1:]
   timestamps = traj[:, 0]
   est_traj = make_dataset_homogenous(est_traj)

   return timestamps, est_traj


def transform_trajectories_save_s3(
        est_input_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_with_timestamp.txt",
        gt_input_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_aligned.txt",
        est_output_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_scaled.txt",
        gt_output_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_scaled.txt",
        scale_path=None,
        scale_estimate=False
):
   df_est = pd.read_csv(est_input_path, sep=" ", header=None, comment="#")
   est_traj = df_est.to_numpy()
   est_timestamp, est_traj = separate_timestamps(est_traj)

   df_gt = pd.read_csv(gt_input_path, sep=" ", header=None, comment="#")
   gt_traj = df_gt.to_numpy()
   gt_timestamp, gt_traj = separate_timestamps(gt_traj)

   gt_traj_trans, est_traj_trans, s = transform_trajectories_mine_final(gt_traj, est_traj, scale_estimate)

   np.savetxt(est_output_path, np.concatenate([est_timestamp.reshape(-1, 1), est_traj_trans[:, :3, :].reshape(-1, 12)], axis=1), fmt='%f', delimiter=' ')
   np.savetxt(gt_output_path, np.concatenate([gt_timestamp.reshape(-1, 1), gt_traj_trans[:, :3, :].reshape(-1, 12)], axis=1), fmt='%f', delimiter=' ')

   with open(scale_path, 'w') as f:
      f.write("#median scale")
      f.write(f"{s}")

def transform_trajectories_display_3d(
        est_input_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_with_timestamp.txt",
        gt_input_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_aligned.txt",
        est_output_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_scaled.txt",
        gt_output_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_scaled.txt",
        scale_estimate=False
):
   df_est = pd.read_csv(est_input_path, sep=" ", header=None, comment="#")
   est_traj = df_est.to_numpy()
   est_timestamp, est_traj = separate_timestamps(est_traj)

   df_gt = pd.read_csv(gt_input_path, sep=" ", header=None, comment="#")
   gt_traj = df_gt.to_numpy()
   gt_timestamp, gt_traj = separate_timestamps(gt_traj)

   gt_traj_trans, est_traj_trans, s = transform_trajectories_mine_final(gt_traj, est_traj, scale_estimate)

   matplotlib.use('TkAgg')
   fig1 = plt.figure()
   ax = fig1.add_subplot(111, projection='3d')
   ax.plot(gt_traj_trans[:, 0, 3], gt_traj_trans[:, 1, 3], gt_traj_trans[:, 2, 3], label="ground_truth")
   ax.plot(est_traj_trans[:, 0, 3], est_traj_trans[:, 1, 3], est_traj_trans[:, 2, 3], label="prediction")
   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
   ax.legend()
   fig1.show()
   plt.show()


def evaluate_trajectories(
        est_input_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_with_timestamp.txt",
        gt_input_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_aligned.txt",
        display=True,
        save=False,
        figure_path=None,
        mean_path=None,
        threshold = None,
        log_scale=False,
        save_error=False,
        save_error_path=None
):
   df_est = pd.read_csv(est_input_path, sep=" ", header=None, comment="#")
   est_traj = df_est.to_numpy()
   est_timestamp, est_traj = separate_timestamps(est_traj)

   df_gt = pd.read_csv(gt_input_path, sep=" ", header=None, comment="#")
   gt_traj = df_gt.to_numpy()
   gt_timestamp, gt_traj = separate_timestamps(gt_traj)

   if threshold is not None:
      est_traj = est_traj[threshold:, :]
      gt_traj = gt_traj[threshold:, :]

   translation_error, rotation_error, translation_error_mean, rotation_error_mean, translation_error_median, rotation_error_median\
      = evaluate(gt_traj, est_traj)

   print(f"Translation error mean: {translation_error_mean}")
   print(f"Rotation error mean: {rotation_error_mean}")

   title_size=18
   label_size=17
   ticks_size=14

   matplotlib.use('TkAgg')
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   ax1.tick_params(axis='both', which='major', labelsize=ticks_size)
   ax1.plot(np.arange(0, translation_error.shape[0], 1), translation_error)
   ax1.set_title("Translation error angle", fontsize=title_size)
   ax1.set_xlabel("Frame", fontsize=label_size)
   ax1.set_ylabel("Angle error (degrees)", fontsize=label_size)

   ax2.plot(np.arange(0, rotation_error.shape[0], 1), rotation_error)
   ax2.tick_params(axis='both', which='major', labelsize=ticks_size)
   ax2.set_title("Rotation error angle", fontsize=title_size)
   ax2.set_xlabel("Frame", fontsize=label_size)
   ax2.set_ylabel("Angle error (degrees)", fontsize=title_size)

   if log_scale:
      ax2.set_yscale('log')

   plt.tight_layout()

   if save:
      assert figure_path is not None
      fig.savefig(figure_path)

      assert mean_path is not None
      with open(mean_path, "w") as f:
         f.write(f"#rotation error mean, translation error mean\n")
         f.write(f"{rotation_error_mean}, {translation_error_mean}\n")
         f.write(f"#rotation error median, translation error median\n")
         f.write(f"{rotation_error_median}, {translation_error_median}\n")

   if display:
      fig.show()
      plt.show()

   if save_error:
      assert save_error_path is not None
      np.savetxt(save_error_path, np.concatenate([translation_error.reshape(-1, 1), rotation_error.reshape(-1, 1)], axis=1),
                 fmt='%f', delimiter=' ', header="# translation error, rotation error", comments="")

def evaluate_placeholder(path, threshold=None, save_error=False, save_error_path=None, log=False):
   evaluate_trajectories(
      est_input_path=path + "/results_se3_aligned_removed.txt",
      gt_input_path=path + "/groundtruth_se3_aligned_removed.txt",
      display=True,
      save=True,
      figure_path=path + "/evaluation.png",
      mean_path=path + "/median.txt",
      log_scale=log,
      threshold=threshold,
      save_error=save_error,
      save_error_path="/home/washindeiru/studia/7_semestr/vo/comparison_final_9" + "/" + save_error_path
   )

def shift_scale_placeholder(path):
   transform_trajectories_save_s3(
      est_input_path=path + "/results_se3_aligned_removed.txt",
      gt_input_path=path + "/groundtruth_se3_aligned_removed.txt",
      est_output_path=path + "/results_se3_aligned_removed_shifted_scaled.txt",
      gt_output_path=path + "/groundtruth_se3_aligned_removed_shifted.txt",
      scale_path=path + "/scale.txt",
      scale_estimate=True
   )

if __name__ == "__main__":
   path = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/kitti_02+2025-01-14_19:35:01"

   # evaluate_placeholder(path, threshold=None, save_error=True, save_error_path="gluestick.txt", log=True)
   shift_scale_placeholder(path)

   # transform_trajectories_save_s3(
   #    est_input_path = path + "/results_se3_aligned_removed.txt",
   #    gt_input_path = path + "/groundtruth_se3_aligned_removed.txt",
   #    est_output_path = path + "/results_se3_aligned_removed_shifted_scaled.txt",
   #    gt_output_path = path + "/groundtruth_se3_aligned_removed_shifted.txt",
   #    scale_path = path + "/scale.txt",
   #    scale_estimate=True
   # )

   # path = "/home/washindeiru/studia/7_semestr/vo/GlueStickOdometry/GlueStick/results/davis_3_calib+2024-12-31_14:01:23"
   #
   # transform_trajectories_save_s3(
   #    est_input_path=path+"/results_se3_aligned_removed.txt",
   #    gt_input_path=path+"/groundtruth_se3_aligned_removed.txt",
   #    est_output_path=path+"/results_se3_aligned_removed_shifted_scaled.txt",
   #    gt_output_path=path+"/groundtruth_se3_aligned_removed_shifted.txt",
   #    scale_estimate=True
   # )

   # transform_trajectories_save_s3(
   #    est_input_path=path+"/stamped_traj_estimate_se3.txt",
   #    gt_input_path=path+"/stamped_groundtruth_aligned_se3.txt",
   #    est_output_path=path+"/stamped_traj_estimate_shifted_scaled_se3.txt",
   #    gt_output_path=path+"/stamped_groundtruth_aligned_shifted_se3.txt",
   #    scale_estimate=True
   # )

   # evaluate_trajectories(
   #    est_input_path=path+"/stamped_traj_estimate_se3.txt",
   #    gt_input_path=path+"/stamped_groundtruth_aligned_se3.txt"
   # )


   # treshold = 1000
   # matplotlib.use('TkAgg')
   #
   # df = pd.read_csv("/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_with_timestamp.txt", sep=" ", header=None)
   # est_traj = df.to_numpy()
   # # remove timestamps
   # est_timestamp, est_traj = separate_timestamps(est_traj)
   # # est_traj = est_traj[:treshold, :, :]
   #
   # df = pd.read_csv("/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_aligned.txt", sep=" ", header=None)
   # gt_traj = df.to_numpy()
   # # remove timestamps
   # gt_timestamp, gt_traj = separate_timestamps(gt_traj)
   # # gt_traj = gt_traj[:treshold, :, :]
   #
   # gt_traj_trans, est_traj_trans, s = transform_trajectories_mine_final(gt_traj, est_traj, True)
   #
   # output_path = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_scaled.txt"
   # np.savetxt(output_path, np.concatenate([est_timestamp.reshape(-1, 1), est_traj_trans[:, :3, :].reshape(-1, 12)], axis=1), fmt='%f', delimiter=' ')
   #
   # output_path_ground_truth = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_scaled.txt"
   # np.savetxt(output_path_ground_truth, np.concatenate([gt_timestamp.reshape(-1, 1), gt_traj_trans[:, :3, :].reshape(-1, 12)], axis=1), fmt='%f', delimiter=' ')
   #
   # translation_error, rotation_error, translation_error_mean, rotation_error_mean = evaluate(gt_traj_trans, est_traj_trans)
   # # translation_error, rotation_error, translation_error_mean, rotation_error_mean = evaluate(gt_traj,
   # #                                                                                           est_traj)
   #
   # print(f"Translation error mean: {translation_error_mean}")
   # print(f"Rotation error mean: {rotation_error_mean}")
   #
   # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
   # ax1.plot(np.arange(0, translation_error.shape[0], 1), translation_error)
   # ax1.set_title("Translation error angle")
   # ax1.set_xlabel("Frame")
   # ax1.set_ylabel("Angle error (degrees)")
   #
   # ax2.plot(np.arange(0, rotation_error.shape[0], 1), rotation_error)
   # ax2.set_title("Rotation error angle")
   # ax2.set_xlabel("Frame")
   # ax2.set_ylabel("Angle error (degrees)")
   # fig.show()
   # plt.show()

   # fig3 = plt.figure()
   # ax4 = fig3.add_subplot(111)
   # ax4.plot(gt_traj_trans[:, 0, 3], gt_traj_trans[:, 1, 3], label="ground_truth")
   # ax4.plot(est_traj_trans[:, 0, 3], est_traj_trans[:, 1, 3], label="prediction")
   # ax4.set_title("Ground truth trajectory")
   # ax4.set_xlabel('X Label')
   # ax4.set_ylabel('Y Label')
   # ax4.legend()
   # fig3.show()
   # plt.show()

   # fig1 = plt.figure()
   # ax = fig1.add_subplot(111, projection='3d')
   # ax.plot(gt_traj_trans[:, 0, 3], gt_traj_trans[:, 1, 3], gt_traj_trans[:, 2, 3], label="ground_truth")
   # ax.set_title("Ground truth trajectory")
   # ax.set_xlabel('X Label')
   # ax.set_ylabel('Y Label')
   # ax.set_zlabel('Z Label')
   # fig1.show()

   # fig2 = plt.figure()
   # ax2 = fig2.add_subplot(111, projection='3d')
   # ax2.plot(est_traj_trans[:, 0, 3], est_traj_trans[:, 1, 3], est_traj_trans[:, 2, 3], label="prediction")
   # ax2.set_title("prediction")
   # ax2.set_xlabel('X Label')
   # ax2.set_ylabel('Y Label')
   # ax2.set_zlabel('Z Label')
   # fig2.show()
   # plt.show()
   #
   # fig1 = plt.figure()
   # ax = fig1.add_subplot(111, projection='3d')
   # ax.plot(gt_traj_trans[:, 0, 3], gt_traj_trans[:, 1, 3], gt_traj_trans[:, 2, 3], label="ground_truth")
   # ax.plot(est_traj_trans[:, 0, 3], est_traj_trans[:, 1, 3], est_traj_trans[:, 2, 3], label="prediction")
   # ax.set_xlabel('X Label')
   # ax.set_ylabel('Y Label')
   # ax.set_zlabel('Z Label')
   # ax.legend()
   # fig1.show()
   # plt.show()
