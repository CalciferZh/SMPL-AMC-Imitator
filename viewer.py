import matplotlib.pyplot as plt
import numpy as np
import motion_parser
from mpl_toolkits.mplot3d import Axes3D


def draw_body(joints):
  fig = plt.figure()
  Axes3D(fig)
  xs, ys, zs = [], [], []
  for joint in joints.values():
    xs.append(joint.coordinate[0])
    ys.append(joint.coordinate[1])
    zs.append(joint.coordinate[2])
  plt.plot(xs, ys, zs, 'b.')

  for joint in joints.values():
    child = joint
    if child.parent is not None:
      parent = child.parent
      xs = [child.coordinate[0], parent.coordinate[0]]
      ys = [child.coordinate[1], parent.coordinate[1]]
      zs = [child.coordinate[2], parent.coordinate[2]]
      plt.plot(xs, ys, zs, 'r')

  plt.show()


if __name__ == '__main__':
  joints = motion_parser.parse_asf('./data/01/01.asf')
  motions = motion_parser.parse_amc('./data/01/01_01.amc')
  for idx in range(0, len(motions), 60):
    joints['root'].set_motion(motions[idx])
    draw_body(joints)
