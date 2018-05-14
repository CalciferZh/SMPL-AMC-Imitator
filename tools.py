import matplotlib.pyplot as plt
import numpy as np
import motion_parser
import smpl_np
import copy
import transforms3d
import copy
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


def set_to_smpl_joints(joints, smpl_J):
  semantic = motion_parser.joint_semantic()
  for k, v in semantic.items():
    joints[v].coordinate = smpl_J[k] / 0.45 * 10


def set_to_smpl_skeleton(joints, smpl):
  to_del = []
  for k in joints.keys():
    if k not in motion_parser.joint_semantic().values():
      to_del.append(k)
  for k in to_del:
    joints[k].parent.children.remove(joints[k])
    del joints[k]
  set_to_smpl_joints(joints, smpl.J)

  for k, v in joints.items():
    if k == 'root':
      continue
    joints[k].direction = v.coordinate - v.parent.coordinate
    joints[k].length = np.linalg.norm(joints[k].direction)
    joints[k].direction /= joints[k].length


def compute_default_R(joints, smpl_J):
  '''Actually we only process legs, i.e. femur and tibia'''
  R = np.stack([np.eye(3) for k in range(24)], axis=0)
  set_to_smpl_joints(joints, smpl_J)
  as_map = motion_parser.asf_smpl_map()

  for bone in ['lfemur', 'rfemur']:
    R[as_map[bone]] = process_femur(joints[bone])

  for bone in ['ltibia', 'rtibia']:
    R[as_map[bone]] = process_tibia(joints[bone])

  return R


def draw_body(joints, xr=(-20, 40), yr=(-10, 50), zr=(-40, 20)):
  fig = plt.figure()
  ax = Axes3D(fig)

  # ax.set_xlim3d(-30, 50)
  # ax.set_ylim3d(0, 30)
  # ax.set_zlim3d(0, 30)

  ax.set_xlim3d(*xr)
  ax.set_ylim3d(*yr)
  ax.set_zlim3d(*zr)

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


def move_skeleton(joints, distance):
  for j in joints.values():
    j.coordinate += distance


def combine_skeletons(roots):
  joints = {}
  for idx, root in enumerate(roots):
    for k, v in root.to_dict().items():
      joints['%s_%d' % (k, idx)] = v
  return joints


def draw_smpl_asf():
  asf_joints = motion_parser.parse_asf('./data/01/01.asf')
  smpl_joints = motion_parser.parse_asf('./data/01/01.asf')
  motions = motion_parser.parse_amc('./data/01/01_01.amc')
  smpl = smpl_np.SMPLModel('./model.pkl')

  set_to_smpl_skeleton(smpl_joints, smpl)

  frame_idx = 180
  asf_joints['root'].set_motion(motions[frame_idx])
  smpl_joints['root'].set_motion(motions[frame_idx])
  move_skeleton(smpl_joints, np.array([0, 0, 30]))

  no_pose = motion_parser.parse_amc('./data/nopose.amc')[0]
  asf_joints_nopose = motion_parser.parse_asf('./data/01/01.asf')
  smpl_joints_nopose = motion_parser.parse_asf('./data/01/01.asf')
  asf_joints_nopose['root'].set_motion(no_pose)
  smpl_joints_nopose['root'].set_motion(no_pose)
  set_to_smpl_skeleton(smpl_joints_nopose, smpl)
  move_skeleton(asf_joints_nopose, np.array([30, 0, 0]))
  move_skeleton(smpl_joints_nopose, np.array([30, 0, 30]))

  combined = combine_skeletons(
    [asf_joints['root'], asf_joints_nopose['root'], smpl_joints['root'], smpl_joints_nopose['root']]
  )

  draw_body(combined)


def extract_R_from_asf_joints(joints, smpl):
  default_R = compute_default_R(joints, smpl.J)
  rotate_R = np.empty([24, 3, 3])

  sa_map = motion_parser.smpl_asf_map()
  for k, v in sa_map.items():
    rotate_R[k] = np.array(joints[v].matrix)
    if joints[v].parent is not None:
      rotate_R[k] = np.dot(np.array(np.linalg.inv(joints[v].parent.matrix)), rotate_R[k])
  R = np.matmul(rotate_R, default_R)
  return R


def align_smpl(joints, smpl):
  default_R = compute_default_R(joints, smpl.J)
  rotate_R = np.empty([24, 3, 3])

  sa_map = motion_parser.smpl_asf_map()
  for k, v in sa_map.items():
    rotate_R[k] = np.array(joints[v].matrix)
    if joints[v].parent is not None:
      rotate_R[k] = np.dot(np.array(np.linalg.inv(joints[v].parent.matrix)), rotate_R[k])
  R = np.matmul(rotate_R, default_R)
  pose = R_to_pose(R)
  verts = smpl.set_params(pose=pose)
  obj_save('./smpl.obj', verts, smpl.faces)


def align_smpl_wrapper():
  joints = motion_parser.parse_asf('./data/01/01.asf')
  motions = motion_parser.parse_amc('./data/01/01_01.amc')
  joints['root'].set_motion(motions[180])
  smpl = smpl_np.SMPLModel('./model.pkl')
  align_smpl(joints, smpl)


def draw_asf_joints_in_motion_wrapper():
  joints = motion_parser.parse_asf('./data/01/01.asf')
  motions = motion_parser.parse_amc('./data/01/01_01.amc')
  joints['root'].set_motion(motions[180])
  draw_body(joints)


def setup_smpl_joints(smpl):
  joints = {}
  for i in range(24):
    joints[i] = motion_parser.SMPLJoints(i)
  for child, parent in smpl.parent.items():
    joints[child].parent = joints[parent]
    joints[parent].children.append(joints[child])
  J = smpl.J / 0.45 * 10
  for j in joints.values():
    j.coordinate = J[j.idx]
  for j in joints.values():
    j.update_info()
  return joints


def smpl_joints_test():
  asf_joints = motion_parser.parse_asf('./data/01/01.asf')
  motions = motion_parser.parse_amc('./data/01/01_01.amc')
  asf_joints['root'].set_motion(motions[180])

  smpl = smpl_np.SMPLModel('./model.pkl')
  smpl_joints = setup_smpl_joints(smpl)
  R = extract_R_from_asf_joints(asf_joints, smpl)
  smpl_joints[0].set_motion(R)

  draw_body(smpl_joints)


if __name__ == '__main__':
  # align_smpl_wrapper()
  # draw_smpl_asf()
  # draw_joints_in_motion_wrapper()
  smpl_joints_test()
