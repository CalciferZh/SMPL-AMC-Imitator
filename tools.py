import matplotlib.pyplot as plt
import numpy as np
import motion_parser
import smpl_np
import copy
import transforms3d
from mpl_toolkits.mplot3d import Axes3D


def compute_rodrigues(x, y):
  theta = np.arccos(np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
  axis = np.cross(x, y)
  return transforms3d.axangles.axangle2mat(axis, theta)


def process_femur(femur):
  hipjoint = femur.parent
  smpl_direction = femur.coordinate - hipjoint.coordinate
  smpl_direction /= np.linalg.norm(smpl_direction)
  asf_direction = np.squeeze(np.array(femur.direction))
  return compute_rodrigues(smpl_direction, asf_direction)


def process_tibia(tibia):
  femur = tibia.parent
  hipjoint = femur.parent
  smpl_femur_dir = femur.coordinate - hipjoint.coordinate
  asf_femur_dir = np.squeeze(np.array(femur.direction))
  smpl_tibia_dir = tibia.coordinate - femur.coordinate
  asf_tibia_dir = np.squeeze(np.array(tibia.direction))
  if not np.allclose(asf_femur_dir, asf_tibia_dir):
    # this case shouldn't happend in CMU dataset
    # so we just leave it here
    print('error: femur and tibia are different!')
    exit()

  return compute_rodrigues(smpl_tibia_dir, smpl_femur_dir)


def set_to_smpl(joints, smpl_J):
  semantic = motion_parser.joint_semantic()
  for k, v in semantic.items():
    joints[v].coordinate = smpl_J[k] / 0.45 * 10


def compute_default_R(joints, smpl_J):
  '''Actually we only process legs, i.e. femur and tibia'''
  R = np.stack([np.eye(3) for k in range(24)], axis=0)
  set_to_smpl(joints, smpl_J)
  as_map = motion_parser.asf_smpl_map()

  for bone in ['lfemur', 'rfemur']:
    R[as_map[bone]] = process_femur(joints[bone])

  # for bone in ['ltibia', 'rtibia']:
  #   R[as_map[bone]] = process_tibia(joints[bone])

  return R


def draw_body(joints):
  fig = plt.figure()
  ax = Axes3D(fig)

  # ax.set_xlim3d(-30, 50)
  # ax.set_ylim3d(0, 30)
  # ax.set_zlim3d(0, 30)

  ax.set_xlim3d(-20, 40)
  ax.set_ylim3d(-30, 30)
  ax.set_zlim3d(-40, 20)

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


def R_to_pose(R):
  pose = np.zeros([24, 3])
  for idx, mat in enumerate(R):
    axis, angle = transforms3d.axangles.mat2axangle(mat)
    axangle = axis / np.linalg.norm(axis) * angle
    pose[idx] = axangle
  return pose


def draw_smpl_asf():
  joints = motion_parser.parse_asf('./data/01/01.asf')
  motions = motion_parser.parse_amc('./data/nopose.amc')
  joints['root'].set_motion(motions[0])

  smpl = smpl_np.SMPLModel('./model.pkl')
  J = smpl.J + np.array([0, 0, 1.5])
  joints_new = motion_parser.parse_asf('./data/01/01.asf')
  set_to_smpl(joints_new, J)

  for k, v in joints_new.items():
    joints[k + '_'] = v

  motions = motion_parser.parse_amc('./data/nopose.amc')
  joints['root'].set_motion(motions[0])
  draw_body(joints)


if __name__ == '__main__':
  # TODO: check all .asf files to see if any unusal default pose
  # IMPORTANT:
  # in smpl, parent is responsible for the bones between parent and all children
  # in asf, child is responsible for the only bone between child and parent

  draw_smpl_asf()

  # smpl = smpl_np.SMPLModel('./model.pkl')

  # joints = motion_parser.parse_asf('./data/01/01.asf')
  # default_R = compute_default_R(joints, smpl.J)

  # frame_idx = 180


  # # motions = motion_parser.parse_amc('./data/nopose.amc')
  # # joints['root'].set_motion(motions[0])

  # motions = motion_parser.parse_amc('./data/01/01_01.amc')
  # joints['root'].set_motion(motions[frame_idx], direction=np.array([-1, -1, -1]))
  # rotate_R = np.empty([24, 3, 3])
  # sa_map = motion_parser.smpl_asf_map()
  # for k, v in sa_map.items():
  #   rotate_R[k] = np.array(joints[v].matrix)
  #   if joints[v].parent is not None:
  #     rotate_R[k] = np.dot(rotate_R[k], np.array(np.linalg.inv(joints[v].parent.matrix)))


  # R = np.matmul(rotate_R, default_R)

  # # semantic = motion_parser.joint_semantic()
  # # jindex = motion_parser.joint_index()



  # # R = np.empty([24, 3, 3])
  # # for i in range(24):
  # #   R[i] = np.eye(3)
  # # for k, v in semantic.items():
  # #   if joints[v].parent is not None:
  # #     idx = jindex[joints[v].parent.name]
  # #     R[idx] = joints[v].default_R

  # # for k, v in semantic.items():
  # #   R[k] = np.dot(R[k], np.array(joints[v].matrix))
  # #   if joints[v].parent is not None:
  # #     R[k] = np.dot(R[k], np.array(np.linalg.inv(joints[v].parent.matrix)))

  # pose = R_to_pose(R)
  # verts = smpl.set_params(pose=pose)
  # obj_save('./smpl.obj', verts, smpl.faces)
