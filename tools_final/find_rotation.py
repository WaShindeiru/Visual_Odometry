import numpy as np
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm

from tools_final.evaluate import make_dataset_homogenous, separate_timestamps


class Rotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotation = nn.Parameter(torch.tensor([[1., 0., 0.],
                                                  [0., 1., 0.],
                                                  [0., 0., 1.],
                                                  [0., 0., 0.]], dtype=torch.float64),
                                     requires_grad=True)

        self.translation = nn.Parameter(torch.tensor([[0.],
                                                     [0.],
                                                     [0.],
                                                      [1.]], dtype=torch.float64),
                                        requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se3_matrix = torch.cat([self.rotation, self.translation], dim=1)
        inverse_se3 = torch.linalg.inv(se3_matrix)

        a = torch.linalg.matmul(inverse_se3, torch.linalg.matmul(x, se3_matrix))
        return a


class RotationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        temp = torch.pow(input - target, 2)
        temp2 = temp[:, :, 3]
        temp3 = torch.linalg.norm(temp2, dim=1)
        return torch.mean(temp3)


def find_rotation(estimated, ground_truth):
    torch.manual_seed(10)

    rotation = Rotation()
    loss_fn = RotationLoss()
    optimizer = torch.optim.Adam(rotation.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(rotation.parameters(), lr=0.01)

    epochs = 100000

    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in tqdm(range(epochs)):
        if epoch > 100 and np.abs(train_loss_values[-1] - train_loss_values[-2]) < 10e-6:
            break

        rotation.train()

        train_prediction = rotation(estimated)

        train_loss = loss_fn(train_prediction, ground_truth)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.inference_mode():
            test_prediction = rotation(estimated)

            test_loss = loss_fn(test_prediction, ground_truth)

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                test_loss_values.append(test_loss.item())
                train_loss_values.append(train_loss.item())
                print(f"Epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}")

    return rotation


def rotate_trajectory(trajectory, rotation):
    trajectory_rotated = np.zeros_like(trajectory)
    rotation_inv = np.linalg.inv(rotation)

    for i in range(trajectory_rotated.shape[0]):
        trajectory_rotated[i, :, :] = rotation_inv @ trajectory[i, :, :] @ rotation
    return trajectory_rotated


def save_se3_timestamps(trajectory, timestamps, path):
    trajectory = trajectory[:, :3, :]
    trajectory = trajectory.reshape(-1, 12)
    timestamps = timestamps.reshape(-1, 1)

    matrix_to_save = np.concatenate([timestamps, trajectory], axis=1)
    np.savetxt(path, matrix_to_save, delimiter=' ')

def find_rotation_and_save(
        estimated_path,
        ground_truth_path,
        estimated_output_path,
        matrix_path
):
    df_est = pd.read_csv(estimated_path, sep=" ", header=None)
    df_gt = pd.read_csv(ground_truth_path, sep=" ", header=None)

    est_traj_np = df_est.to_numpy()
    gt_traj_np = df_gt.to_numpy()

    est_timestamps, est_traj_np = separate_timestamps(est_traj_np)
    gt_timestamps, gt_traj_np = separate_timestamps(gt_traj_np)

    est_traj = torch.from_numpy(est_traj_np)
    gt_traj = torch.from_numpy(gt_traj_np)

    rotation = find_rotation(est_traj, gt_traj)

    rotation_matrix = rotation.state_dict()['rotation'].detach().numpy()
    single_vector = rotation.state_dict()['translation'].detach().numpy()

    rotation_matrix_se3 = np.concatenate([rotation_matrix, single_vector], axis=1)

    np.savetxt(matrix_path, rotation_matrix_se3, delimiter=' ')
    est_traj_rotated = rotate_trajectory(est_traj_np, rotation_matrix_se3)

    save_se3_timestamps(est_traj_rotated, est_timestamps, estimated_output_path)


if __name__ == "__main__":
    path = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/kitti_02+2025-01-14_19:35:01"
    find_rotation_and_save(
        estimated_path=path+"/results_se3_aligned_removed_shifted_scaled.txt",
        ground_truth_path=path+"/groundtruth_se3_aligned_removed_shifted.txt",
        estimated_output_path=path+"/results_se3_aligned_removed_shifted_scaled_rotated.txt",
        matrix_path=path+"/rotation_matrix.txt"
    )