from os import mkdir
from os.path import isdir
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import torch

from TartanVO import TartanVO
from Tartanvo.Datasets.utils import dataset_intrinsics, visflow
from Datasets.tartanTrajFlowDatasetBetter import *
from Datasets.transformation import *
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow
from tools import *

from tools import save_as_s3_with_timestamp_experimental


class TantanOdometry():
   def __init__(self, image_height=448, image_width=640, model_name="tartanvo_1914.pkl", datastr='davis_3_calib',
                test_dir="/media/washindeiru/New Volume/backup/windowsBackup16-01-2025/g/odom_files/DEVOv2/DEVO/data/indoor_forward_3_davis_with_gt/img_short"
                , batch_size=1, worker_num=6, ground_truth_path="/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/poses.txt"):

      self.model_name = model_name
      self.datastr = datastr

      self.test_dir = test_dir
      self.sequence_name = datastr

      current_datetime = datetime.now()

      datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

      self.output_name = f"{self.sequence_name}+{datetime_string}"

      self.tartan = TartanVO(model_name)
      # self.tartan = torch.compile(self.tartan)

      self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
      # if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
      #    focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

      transform = Compose([CropCenter((image_height, image_width)), DownscaleFlow(), ToTensor()])

      self.testDataset = TrajFolderDatasetBetter(test_dir, transform=transform, focalx=self.focalx, focaly=self.focaly,
                                                 centerx=self.centerx, centery=self.centery, rgb=True)
      self.testDataloader = DataLoader(self.testDataset, batch_size=batch_size,
                                  shuffle=False, num_workers=worker_num)
      self.testDataiter = iter(self.testDataloader)

      ## delete this
      # poses = pd.read_csv(ground_truth_path, sep=" ", header=None)
      #
      # self.ground_truth = np.zeros((len(poses), 3, 4))
      # for i in range(len(poses)):
      #    self.ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))

      self.output_directory = Path("results") / self.output_name
      self.output_directory.mkdir(parents=True, exist_ok=True)


   def visual_odometry(self, subset: int = None):
      if subset is None:
         num_frames = len(self.testDataset) + 1
      else:
         num_frames = subset

      save_flow = True

      transformation_matrix = np.eye(4, dtype=np.float64)
      # transformation_matrix = np.array([[9.492627669324508188e-01, -3.142715022164684902e-01, -1.155950735766076570e-02, -2.853500841717197094e+06],
      #                                   [3.144719965220956759e-01, 9.489064403320720542e-01, 2.615207257030869453e-02, 4.667424270940385759e+06],
      #                                   [2.750039846005262502e-03, -2.846033012668491846e-02, 9.995911398616563748e-01, 3.268261645953029860e+06]])

      trajectory = np.zeros((num_frames, 3, 4))
      trajectory[0] = transformation_matrix[:3, :]

      motionlist = []
      timestamptContainer = list()
      temp = self.testDataset.getFirstTimestamp()
      # temp = (temp[:10] + "." + temp[10:])
      timestamptContainer.append(float(temp))

      # testname = self.datastr + '_' + self.model_name.split('.')[0]
      # if save_flow:
      #    flowdir = 'results/' + testname + '_flow'
      #    if not isdir(flowdir):
      #       mkdir(flowdir)
      #    flowcount = 0

      flowcount = 0

      for i in tqdm(range(num_frames - 1)):
         sample = next(self.testDataiter)

      # while True:
      #    try:
      #       sample = next(self.testDataiter)
      #    except StopIteration:
      #       break

         motions, flow = self.tartan.test_batch(sample)
         SE3motion = se2SE_better(motions)

         # for k in range(flow.shape[0]):
         #    flowk = flow[k].transpose(1, 2, 0)
         #    # np.save(self.output_name + '/' + str(flowcount).zfill(6) + '.npy', flowk)
         #    flow_vis = visflow(flowk)
         #    cv2.imwrite("./results/" + self.output_name + '/' + str(flowcount) + '.png', flow_vis)
         #    flowcount += 1

         motionlist.extend(motions)
         temp = sample['timestamp'][0]
         # temp = temp[:10] + "." + temp[10:]
         timestamptContainer.append(float(temp))

         transformation_matrix = transformation_matrix @ np.linalg.inv(SE3motion)
         trajectory[i+1, :, :] = transformation_matrix[:3, :]


      # poselist = ses2poses_quat(np.array(motionlist))


      # calculate ATE, RPE, KITTI-RPE
      # if pose_file.endswith('.txt'):
      # evaluator = TartanAirEvaluator()
      # results = evaluator.evaluate_one_trajectory(poses, poselist, scale=True,
      #                                             kittitype=(datastr == 'kitti'))
      # if datastr == 'euroc':
      #    print("==> ATE: %.4f" % (results['ate_score']))
      # else:
      #    print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" % (
      #    results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
      #
      # # save results and visualization
      # plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/' + testname + '.png',
      #           title='ATE %.4f' % (results['ate_score']))
      # np.savetxt('results/' + testname + '.txt', results['est_aligned'])
      # else:


      # np.savetxt('results/' + testname + '.txt', poselist)

      # plot_path_with_matrix("KITTI_10", self.ground_truth[:subset, :, :], trajectory)

      save_3d_plot(self.output_name, trajectory)
      save_as_s3_with_timestamp_experimental(self.output_name, trajectory, timestamptContainer)


if __name__ == "__main__":
   tartan = TantanOdometry()
   tartan.visual_odometry(subset=None)
