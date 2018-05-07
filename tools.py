import matplotlib.pyplot as plt
import numpy as np
import motion_parser
import smpl_np as smpl
import copy
import transforms3d
from mpl_toolkits.mplot3d import Axes3D


def compute_rodrigues(x, y):
  theta = np.arccos(np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
  axis = np.cross(x, y)
  return transforms3d.axangles.axangle2mat(axis, theta)


def compute_smpl_direction(joints):
  R = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), (24, 3, 3))
  _, _, J = smpl.smpl_model('./model.pkl', R)
  semantic = motion_parser.joint_semantic()
  for k, v in semantic.items():
    joints[v].coordinate = J[k]
  for k, v in semantic.items():
    child_joint = joints[v]
    parent_joint = child_joint.parent
    if parent_joint is None:
      continue
    smpl_direction = child_joint.coordinate - parent_joint.coordinate
    smpl_direction /= np.linalg.norm(smpl_direction)
    asf_direction = np.squeeze(np.array(child_joint.direction))
    child_joint.default_R = compute_rodrigues(smpl_direction, asf_direction)


def draw_body(joints):
  fig = plt.figure()
  ax = Axes3D(fig)

  ax.set_xlim3d(-30, 50)
  ax.set_ylim3d(0, 30)
  ax.set_zlim3d(0, 30)

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


def obj_save(path, vertices, faces=None):
  with open(path, 'w') as fp:
    for v in vertices:
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    if faces is not None:
      for f in faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


if __name__ == '__main__':
  joints = motion_parser.parse_asf('./data/01/01.asf')
  compute_smpl_direction(joints)
  # R = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), (24, 3, 3))
  # _, _, J = smpl.smpl_model('./model.pkl', R)
  # J *= 39
  # semantic = motion_parser.joint_semantic()
  # for k, v in semantic.items():
  #   joints[v].coordinate = J[k]
  # draw_body(joints)
  # motions = motion_parser.parse_amc('./data/fake.amc')
  # # motions = motion_parser.parse_amc('./data/01/01_01.amc')
  # # motions = motion_parser.parse_amc('./data/nopose.amc')
  # semantic = motion_parser.joint_semantic()
  # for idx in range(0, len(motions), 60):
  #   rem = copy.deepcopy(motions[idx])
  #   joints['root'].set_motion(motions[idx], direction=np.array([-1, -1, -1]))

  #   R = np.empty([24, 3, 3])
  #   for k, v in semantic.items():
  #     R[k] = joints[v].matrix
  #     if joints[v].parent is not None:
  #       R[k] = R[k] * np.linalg.inv(joints[v].parent.matrix)

  #   verts, faces = smpl.smpl_model('./model.pkl', R)

  #   obj_save('smpl.obj', verts, faces)

  #   joints['root'].set_motion(rem, direction=np.array([1, 1, 1]))
  #   draw_body(joints)
  #   break
