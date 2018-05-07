import matplotlib.pyplot as plt
import numpy as np
import motion_parser
import smpl_np as smpl
from mpl_toolkits.mplot3d import Axes3D


def draw_body(joints):
  fig = plt.figure()
  ax = Axes3D(fig)

  ax.set_xlim3d(-30, 50)
  ax.set_ylim3d(-30, 30)
  ax.set_zlim3d(-30, 30)

  xs, ys, zs = [], [], []
  for joint in joints.values():
    if joint.coordinate is None:
      continue
    xs.append(joint.coordinate[0])
    ys.append(joint.coordinate[1])
    zs.append(joint.coordinate[2])
  plt.plot(zs, xs, ys, 'b.')

  for joint in joints.values():
    if joint.coordinate is None:
      continue
    child = joint
    if child.parent is not None:
      parent = child.parent
      xs = [child.coordinate[0], parent.coordinate[0]]
      ys = [child.coordinate[1], parent.coordinate[1]]
      zs = [child.coordinate[2], parent.coordinate[2]]
      plt.plot(zs, xs, ys, 'r')
  plt.show()


def set_joints_smpl(joints):
  _, _, J = smpl.simple_smpl(want_J=True)
  J *= 39.37
  semantic = motion_parser.joint_semantic()
  for k, v in semantic.items():
    joints[v].coordinate = J[k]

if __name__ == '__main__':
  joints = motion_parser.parse_asf('./data/01/01.asf')
  set_joints_smpl(joints)
  draw_body(joints)
  # motions = motion_parser.parse_amc('./data/01/01_01.amc')
  # for idx in range(0, len(motions), 60):
  #   joints['root'].set_motion(motions[idx])
  #   draw_body(joints)
