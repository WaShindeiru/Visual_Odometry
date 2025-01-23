import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer


class TrajFolderDatasetBetter(Dataset):

   def __init__(self, imgfolder, rgb, posefile=None, transform=None,
                focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

      self.rgb = rgb
      self.files_ = listdir(imgfolder)

      self.rgbfiles = [ff for ff in self.files_ if (ff.endswith('.png') or ff.endswith('.jpg'))]
      # self.rgbfiles.sort()
      self.rgbfiles.sort(key=lambda x: int(x.split(".")[0]))
      self.temp = self.rgbfiles
      self.rgbfiles = [imgfolder + '/' + ff for ff in self.rgbfiles]
      self.imgfolder = imgfolder

      self.timestamps = [ff.split(".")[0] for ff in self.temp]
      # self.timestamps.sort()

      print('Find {} image files_ in {}'.format(len(self.rgbfiles), imgfolder))

      if posefile is not None and posefile != "":
         poselist = np.loadtxt(posefile).astype(np.float32)
         assert (poselist.shape[1] == 7)  # position + quaternion
         poses = pos_quats2SEs(poselist)
         self.matrix = pose2motion(poses)
         self.motions = SEs2ses(self.matrix).astype(np.float32)
         # self.motions = self.motions / self.pose_std
         assert (len(self.motions) == len(self.rgbfiles)) - 1
      else:
         self.motions = None

      self.N = len(self.rgbfiles) - 1

      # self.N = len(self.lines)
      self.transform = transform
      self.focalx = focalx
      self.focaly = focaly
      self.centerx = centerx
      self.centery = centery

   def __len__(self):
      return self.N

   def __getitem__(self, idx):
      imgfile1 = self.rgbfiles[idx].strip()
      imgfile2 = self.rgbfiles[idx + 1].strip()

      if self.rgb:
         img1 = cv2.imread(imgfile1)
         img2 = cv2.imread(imgfile2)
      else:
         img1 = cv2.imread(imgfile1, cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(imgfile2, cv2.IMREAD_GRAYSCALE)

      res = {'img1': img1, 'img2': img2}

      if self.rgb:
         h, w, _ = img1.shape
      else:
         h, w = img1.shape

      intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
      res['intrinsic'] = intrinsicLayer

      if self.transform:
         res = self.transform(res)

      res['timestamp'] = self.timestamps[idx + 1]

      if self.motions is None:
         return res
      else:
         res['motion'] = self.motions[idx]
         return res


   def getFirstTimestamp(self):
      return self.timestamps[0]