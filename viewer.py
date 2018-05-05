import matplotlib.pyplot as plt
import numpy as np
import motion_parser
from mpl_toolkits.mplot3d import Axes3D


def set_bone_traverse(current_bone, bones):
  for bone_name in current_bone.children:
    bone = bones[bone_name]
    bone.coordinate = bones[bone.parent].coordinate - bone.direction * bone.length
    set_bone_traverse(bone, bones)


def draw_body(bones):
  root_bone = bones['root']
  root_bone.coordinate = np.zeros(3)
  set_bone_traverse(root_bone, bones)

  fig = plt.figure()
  Axes3D(fig)
  xs, ys, zs = [], [], []
  for bone in bones.values():
    xs.append(bone.coordinate[0])
    ys.append(bone.coordinate[1])
    zs.append(bone.coordinate[2])
  plt.plot(xs, ys, zs, 'b.')

  for bone in bones.values():
    child = bone
    if child.parent is not None:
      parent = bones[child.parent]
      xs = [child.coordinate[0], parent.coordinate[0]]
      ys = [child.coordinate[1], parent.coordinate[1]]
      zs = [child.coordinate[2], parent.coordinate[2]]
      plt.plot(xs, ys, zs, 'r')

  plt.show()


if __name__ == '__main__':
  bones = motion_parser.parse_asf('./data/01/01.asf')
  for bone in bones.values():
    bone.pretty_print()
  draw_body(bones)
