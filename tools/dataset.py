import pandas as pd
import numpy as np
import os
import cv2

from matplotlib import pyplot as plt

class DatasetHandler():
	def __init__(self, sequence_name):

		self.seq_dir = f"../dataset/{sequence_name}"
		self.poses_dir = f"../dataset/{sequence_name}/poses.txt"
		poses = pd.read_csv(self.poses_dir, sep=" ", header=None)

		self.left_image_path_list = sorted(os.listdir(self.seq_dir + "/image_0"), key=lambda x: int(x.split(".")[0]))
		self.num_frames = len(self.left_image_path_list)

		calib = pd.read_csv(self.seq_dir + "/calib.txt", sep=" ", header=None, index_col=0)
		self.projectionMatrix_left = np.array(calib.loc["P0:"]).reshape((3, 4))
		self.projectionMatrix_right = np.array(calib.loc["P1:"]).reshape((3, 4))

		self.intrinsic_matrix, self.extrinsic_matrix = DatasetHandler.decompose_projection_matrix(self.projectionMatrix_left)

		print(f"Projection matrix: \n{self.projectionMatrix_left}")
		print(f"Intrinsic matrix: \n{self.intrinsic_matrix}")
		print(f"Extrinsic matrix: \n{self.extrinsic_matrix}")

		self.ground_truth = np.zeros((len(poses), 3, 4))
		for i in range(len(poses)):
			self.ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))

		print(self.ground_truth.shape)

		self.reset_frames()

		self.first_image_left = cv2.imread(self.seq_dir + "/image_0/" + self.left_image_path_list[0], 0)
		self.second_image_left = cv2.imread(self.seq_dir + "/image_0/" + self.left_image_path_list[1], 0)

		self.image_height = self.first_image_left.shape[0]
		self.image_width = self.first_image_left.shape[1]

	@staticmethod
	def decompose_projection_matrix(projection_matrix):
		k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(projection_matrix)
		t1 = t1 / t1[3]

		transform = np.eye(4, dtype=np.float64)
		transform[:3, :3] = r1
		transform[:3, 3] = t1[:3, 0]

		return k1, transform


	def reset_frames(self):
		self.images_left = (cv2.imread(self.seq_dir + "/image_0/" + name, 0) for name in self.left_image_path_list)



if __name__ == "__main__":
	Dataset = DatasetHandler("00")

	plt.figure()
	plt.imshow(Dataset.first_image_left, 'gray')
	plt.show()

	plt.figure()
	plt.imshow(Dataset.second_image_left, 'gray')
	plt.show()