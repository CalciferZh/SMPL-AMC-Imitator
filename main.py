import reader
from smpl_np import SMPLModel
from skeleton import *
from vistool import *
from transfer import *


def main():
  asf_joints = reader.parse_asf('./data/01/01.asf')
  asf_joints['root'].reset_pose()

  smpl = SMPLModel('./model.pkl')
  smpl_joints = setup_smpl_joints(smpl)

  R = align_smpl_asf(asf_joints, smpl_joints)

  smpl_joints[0].set_motion(R)

  move_skeleton(smpl_joints, [40, 0, 0])

  combined = combine_skeletons([
    smpl_joints,
    asf_joints
  ])

  draw_body(combined)



if __name__ == '__main__':
  # align_smpl_wrapper()
  # draw_smpl_asf()
  # draw_joints_in_motion_wrapper()
  # smpl_joints_test()
  # test()
  main()



# def R_to_pose(R):
#   pose = np.zeros([24, 3])
#   for idx, mat in enumerate(R):
#     axis, angle = transforms3d.axangles.mat2axangle(mat)
#     axangle = axis / np.linalg.norm(axis) * angle
#     pose[idx] = axangle
#   return pose


# def draw_smpl_asf():
#   motions = motion_parser.parse_amc('./data/01/01_01.amc')
#   frame_idx = 180

#   # no_pose = motion_parser.parse_amc('./data/nopose.amc')[0]

#   asf_joints = motion_parser.parse_asf('./data/01/01.asf')
#   asf_joints['root'].set_motion(motions[frame_idx])

#   # asf_joints_nopose = motion_parser.parse_asf('./data/01/01.asf')
#   # asf_joints_nopose['root'].set_motion(no_pose)
#   # move_skeleton(asf_joints_nopose, np.array([0, 0, -30]))

#   smpl = smpl_np.SMPLModel('./model.pkl')

#   smpl_joints = setup_smpl_joints(smpl)
#   R = extract_R_from_asf_joints(asf_joints, smpl)
#   smpl_joints[0].set_motion(R)
#   # move_skeleton(smpl_joints, np.array([40, 0, 0]))

#   # smpl_joints_nopose = setup_smpl_joints(smpl)
#   # R = extract_R_from_asf_joints(asf_joints_nopose, smpl)
#   # smpl_joints_nopose[0].set_motion(R)
#   # move_skeleton(smpl_joints_nopose, np.array([40, 0, -30]))

#   # combined = combine_skeletons(
#   #   [asf_joints['root'], asf_joints_nopose['root'], smpl_joints[0], smpl_joints_nopose[0]]
#   # )

#   # draw_body(combined, xr=(20, -40), yr=(-20, 60))


# def extract_R_from_asf_joints(joints, smpl):
#   smpl_joints = setup_smpl_joints(smpl)
#   default_R = compute_default_R(joints, smpl_joints)
#   rotate_R = np.empty([24, 3, 3])

#   sa_map = motion_parser.smpl_asf_map()
#   for k, v in sa_map.items():
#     rotate_R[k] = np.array(joints[v].matrix)
#     if joints[v].parent is not None:
#       rotate_R[k] = np.dot(np.array(np.linalg.inv(joints[v].parent.matrix)), rotate_R[k])
#   R = np.matmul(rotate_R, default_R)
#   # R = default_R
#   # R = rotate_R
#   return R


# def align_smpl(joints, smpl):
#   default_R = compute_default_R(joints, smpl.J)
#   rotate_R = np.empty([24, 3, 3])

#   sa_map = motion_parser.smpl_asf_map()
#   for k, v in sa_map.items():
#     rotate_R[k] = np.array(joints[v].matrix)
#     if joints[v].parent is not None:
#       rotate_R[k] = np.dot(np.array(np.linalg.inv(joints[v].parent.matrix)), rotate_R[k])
#   R = np.matmul(rotate_R, default_R)
#   pose = R_to_pose(R)
#   verts = smpl.set_params(pose=pose)
#   obj_save('./smpl.obj', verts, smpl.faces)


# def align_smpl_wrapper():
#   joints = motion_parser.parse_asf('./data/01/01.asf')
#   motions = motion_parser.parse_amc('./data/01/01_01.amc')
#   joints['root'].set_motion(motions[180])
#   smpl = smpl_np.SMPLModel('./model.pkl')
#   align_smpl(joints, smpl)


# def draw_asf_joints_in_motion_wrapper():
#   joints = motion_parser.parse_asf('./data/01/01.asf')
#   motions = motion_parser.parse_amc('./data/01/01_01.amc')
#   joints['root'].set_motion(motions[180])
#   draw_body(joints)


# def smpl_joints_test():
#   asf_joints = motion_parser.parse_asf('./data/01/01.asf')
#   motions = motion_parser.parse_amc('./data/01/01_01.amc')
#   asf_joints['root'].set_motion(motions[180])

#   smpl = smpl_np.SMPLModel('./model.pkl')
#   smpl_joints = setup_smpl_joints(smpl)
#   R = extract_R_from_asf_joints(asf_joints, smpl)
#   smpl_joints[0].set_motion(R)

#   smpl_joints_nopose = setup_smpl_joints(smpl)
#   move_skeleton(smpl_joints_nopose, np.array([30, 0, 30]))

#   combined = combine_skeletons([smpl_joints[0], smpl_joints_nopose[0]])

#   draw_body(combined)


# def test():
  # motions = motion_parser.parse_amc('./data/01/01_01.amc')
  # frame_idx = 180

  # no_pose = motion_parser.parse_amc('./data/nopose.amc')[0]

  # asf_joints = motion_parser.parse_asf('./data/01/01.asf')
  # asf_joints['root'].set_motion(motions[frame_idx])

  # asf_joints_nopose = motion_parser.parse_asf('./data/01/01.asf')
  # asf_joints_nopose['root'].set_motion(no_pose)
  # for j in asf_joints_nopose.values():
  #   print(j.name, np.squeeze(np.array(j.direction)))
  # move_skeleton(asf_joints_nopose, np.array([0, 0, -30]))

  # smpl = smpl_np.SMPLModel('./model.pkl')

  # smpl_joints = setup_smpl_joints(smpl)
  # R = extract_R_from_asf_joints(asf_joints, smpl)
  # smpl_joints[0].set_motion(R)
  # move_skeleton(smpl_joints, np.array([40, 0, 0]))

  # smpl_joints_nopose = setup_smpl_joints(smpl)
  # R = extract_R_from_asf_joints(asf_joints_nopose, smpl)
  # smpl_joints_nopose[0].set_motion(R)
  # for j in smpl_joints_nopose.values():
  #   if j.parent is None:
  #     print(None)
  #   else:
  #     print(j.idx, j.coordinate - j.parent.coordinate)
  # move_skeleton(smpl_joints_nopose, np.array([40, 0, -30]))

  # combined = combine_skeletons(
  #   [asf_joints['root'], asf_joints_nopose['root'], smpl_joints[0], smpl_joints_nopose[0]]
  # )

  # draw_body(combined, xr=(20, -40), yr=(-20, 60))

  # combined = combine_skeletons(
  #   [asf_joints_nopose['root'], smpl_joints_nopose[0]]
  # )

  # draw_body(combined, xr=(20, -40), yr=(-20, 60))



