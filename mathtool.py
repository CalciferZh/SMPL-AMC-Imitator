import transforms3d
import numpy as np


def compute_rodrigues(x, y):
  ''' y = Rx '''
  theta = np.arccos(np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
  axis = np.cross(x, y)
  return transforms3d.axangles.axangle2mat(axis, theta)
