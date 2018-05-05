import matplotlib.pyplot as plt
import numpy as np
import motion_parser
from mpl_toolkits.mplot3d import Axes3D


def set_bone_traverse(parent_joint):
  for child_joint in parent_joint.children:
    child_joint.coordinate = parent_joint.coordinate - child_joint.direction * child_joint.length
    set_bone_traverse(child_joint)


def draw_body(joints):
  root_joint = joints['root']
  set_bone_traverse(root_joint)

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
  joints['root'].coordinate = np.zeros(3)
  draw_body(joints)
