import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_quat_timestamp_3d(path="/home/washindeiru/studia/7_semestr/vo/papers/DEVOv2/DEVO/data/indoor_forward_9_davis_with_gt/groundtruth.txt"):
    df = pd.read_csv(path, sep=" ", header=None, comment="#")
    matrix = df.to_numpy()

    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(matrix[:, 1], matrix[:, 2], matrix[:, 3], color='g', label="trajektoria referencyjna")
    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.set_zlabel('z (m)', fontsize=14)
    ax.legend(fontsize=16)

    ax.scatter(matrix[0, 1], matrix[0, 2], matrix[0, 3], color='blue', label='start', s=50, zorder=5)
    ax.text(matrix[0, 1], matrix[0, 2], matrix[0, 3], 'Początek')

    # Mark the end point
    ax.scatter(matrix[-1, 1], matrix[-1, 2], matrix[-1, 3], color='red', label='end', s=50, zorder=5)
    ax.text(matrix[-1, 1], matrix[-1, 2], matrix[-1, 3], 'Koniec')

    plt.tight_layout()

    plt.show()

def plot_quat_timestamp_2d(path, save=False, output_path=None):
    df = pd.read_csv(path, sep=" ", header=None, comment="#")
    matrix = df.to_numpy()

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(7, 6))
    plt.plot(matrix[:, 1], matrix[:, 2], color='g', label="trajektoria referencyjna")
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('y (m)', fontsize=14)
    plt.legend(fontsize=16)

    plt.scatter(matrix[0, 1], matrix[0, 2], color='blue', label='start', zorder=5)
    plt.text(matrix[0, 1], matrix[0, 2], 'Początek', ha='right', va='bottom')

    # Mark the end point
    plt.scatter(matrix[-1, 1], matrix[-1, 2], color='red', label='end', zorder=5)
    plt.text(matrix[-1, 1], matrix[-1, 2], 'Koniec', ha='left', va='bottom')

    plt.tight_layout()

    if save:
        assert output_path is not None
        plt.savefig(output_path)

    plt.show()


def plot_s3_timestamp(path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_with_timestamp.txt"):

    df = pd.read_csv(path, sep=" ", header=None, comment="#")
    matrix = df.to_numpy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(matrix[:, 4], matrix[:, 8], matrix[:, 12])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()


def plot_quat_timestamp(path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_with_timestamp.txt"):

    df = pd.read_csv(path, sep=",", header=None, comment="#")
    matrix = df.to_numpy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(matrix[:, 1], matrix[:, 2], matrix[:, 3])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()



def plot_s3_timestamp_estimated_groundtruth(
        groundtruth_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_aligned.txt",
        estimated_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_s3_scaled.txt"
    ):

    df = pd.read_csv(estimated_path, sep=" ", header=None, comment="#")
    prediction = df.to_numpy()

    df = pd.read_csv(groundtruth_path, sep=" ", header=None, comment="#")
    groundtruth = df.to_numpy()

    df = pd.read_csv(groundtruth_path, sep=" ", header=None, comment="#")
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.plot(groundtruth[:, 4], groundtruth[:, 8], groundtruth[:, 12])
    ax2.plot(prediction[:, 4], prediction[:, 8], prediction[:, 12])
    plt.show()


def plot_s3_timestamp_estimated_groundtruth_one_ax(
        groundtruth_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_scaled.txt",
        estimated_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_se3_scaled_rotated.txt",
        save=False,
        figure_path = None
    ):
    df = pd.read_csv(estimated_path, sep=" ", header=None, comment="#")
    prediction = df.to_numpy()

    df = pd.read_csv(groundtruth_path, sep=" ", header=None, comment="#")
    groundtruth = df.to_numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot(groundtruth[:, 4], groundtruth[:, 8], groundtruth[:, 12], label="ground_truth", color='g')
    ax1.plot(prediction[:, 4], prediction[:, 8], prediction[:, 12], label="estimated", color='orange')
    ax1.legend(loc="upper left", fontsize=15)
    ax1.set_xlabel('x (m)', fontsize=13)
    ax1.set_ylabel('z (m)', fontsize=13)
    ax1.set_zlabel('y (m)', fontsize=13)
    plt.tight_layout()

    if save:
        assert figure_path is not None
        plt.savefig(figure_path)

    plt.show()


def plot_s3_timestamp_estimated_groundtruth_2d(
        groundtruth_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/ground_truth_scaled.txt",
        estimated_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/davis_3_calib+2024-12-30_10:56:48/results_se3_scaled_rotated.txt",
        save=False,
        figure_path = None
    ):
    plt.rcParams.update({'font.size': 12})
    df = pd.read_csv(estimated_path, sep=" ", header=None, comment="#")
    prediction = df.to_numpy()

    df = pd.read_csv(groundtruth_path, sep=" ", header=None, comment="#")
    groundtruth = df.to_numpy()

    fig = plt.figure()
    plt.plot(groundtruth[:, 4], groundtruth[:, 12], label="ground_truth", color='g')
    plt.plot(prediction[:, 4], prediction[:, 12], label="estimated", color='orange')
    plt.xlabel('x (m)', fontsize=13)
    plt.ylabel('y (m)', fontsize=13)
    plt.tight_layout()
    plt.legend(loc="upper left", fontsize=16)

    if save:
        assert figure_path is not None
        plt.savefig(figure_path)

    plt.show()


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    plot_quat_timestamp_3d("/media/washindeiru/Hard Disc/odom_files/DEVOv2/DEVO/data/indoor_forward_9_davis_with_gt/groundtruth.txt")
    # plot_quat_timestamp_2d(
    #     path="/media/washindeiru/Hard Disc/odom_files/DEVOv2/DEVO/data/indoor_forward_9_davis_with_gt/groundtruth.txt",
    #     save=True,
    #     output_path="/media/washindeiru/Hard Disc/odom_files/DEVOv2/DEVO/data/indoor_forward_9_davis_with_gt/evaluate_mine/groundtruth_2d_new.png"
    # )
    # plot_quat_timestamp_2d(path="/home/washindeiru/studia/7_semestr/vo/papers/DEVOv2/DEVO/data/indoor_forward_9_davis_with_gt/groundtruth.txt",
    #                        save=True,
    #                        output_path="/home/washindeiru/studia/7_semestr/vo/papers/DEVOv2/DEVO/data/indoor_forward_9_davis_with_gt/evaluate_mine/groundtruth_2d.png")

    path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/kitti_02+2025-01-14_19:35:01"

    # plot_quat_timestamp(path)

    # plot_s3_timestamp_estimated_groundtruth_one_ax(
    #    groundtruth_path=path+"/groundtruth_se3_aligned_removed_shifted.txt",
    #    estimated_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt"
    # )

    # plot_s3_timestamp_estimated_groundtruth_2d(
    #     groundtruth_path=path+"/groundtruth_se3_aligned_removed.txt",
    #     estimated_path=path+"/results_se3_aligned_removed.txt"
    # )

    # plot_s3_timestamp_estimated_groundtruth_2d(
    #     groundtruth_path=path+"/groundtruth_se3_aligned_removed_shifted.txt",
    #     estimated_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt",
    #     save=True,
    #     figure_path=path+"/comparison_2d.png"
    # )

    # plot_s3_timestamp_estimated_groundtruth_one_ax(
    #    groundtruth_path="/home/washindeiru/studia/7_semestr/vo/papers/DEVOv2/DEVO/data/indoor_forward_3_davis_with_gt/groundtruth_se3.txt",
    #    estimated_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt"
    # )

    # plot_s3_timestamp_estimated_groundtruth_one_ax(
    #    groundtruth_path=path+"/groundtruth_se3_aligned_removed_shifted.txt",
    #    estimated_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt"
    # )


    # path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/SuperGlue/results/davis_3_calib+2025-01-01_12:59:47"
    # plot_s3_timestamp_estimated_groundtruth_2d(
    #     estimated_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt",
    #     groundtruth_path=path+"/groundtruth_se3_aligned_removed_shifted.txt",
    #     save=True,
    #     figure_path=path+"/comparison_2d.png"
    # )
    # plot_s3_timestamp_estimated_groundtruth_one_ax(
    #     estimated_path=path+"/stamped_traj_estimate_se3.txt",
    #     groundtruth_path=path+"/stamped_groundtruth_aligned_se3.txt"
    # )
    # plot_s3_timestamp_estimated_groundtruth_one_ax(
    #     estimated_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt",
    #     groundtruth_path=path+"/groundtruth_se3_aligned_removed_shifted.txt"
    # )

    # path = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/Tartanvo/results/davis_3_calib+2025-01-01_11:29:05"
    #
    # plot_s3_timestamp_estimated_groundtruth_one_ax(
    #     estimated_path=path + "/results_se3_aligned_removed_shifted_scaled_rotated.txt",
    #     groundtruth_path=path + "/groundtruth_se3_aligned_removed_shifted.txt",
    #     save=True,
    #     figure_path=path + "/comparison.png"
    # )