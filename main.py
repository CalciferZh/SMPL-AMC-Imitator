import reader
from smpl_np import SMPLModel
from skeleton import *
from vistool import *
from transfer import *


def draw_nopose():
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

  print('===================== ASF =======================')
  for j in asf_joints.values():
    print(j.name, j.direction)

  print('===================== SMPL ======================')
  for j in smpl_joints.values():
    if j.parent is None:
      print(j.idx, None)
    else:
      print(j.idx, j.coordinate - j.parent.coordinate, j.to_parent)


def draw_four():
  asf_joints = reader.parse_asf('./data/01/01.asf')
  asf_joints['root'].reset_pose()

  smpl = SMPLModel('./model.pkl')
  smpl_joints = setup_smpl_joints(smpl)

  R = align_smpl_asf(asf_joints, smpl_joints)

  smpl_joints[0].set_motion(R)

  move_skeleton(smpl_joints, [40, 0, 0])



if __name__ == '__main__':
  # align_smpl_wrapper()
  # draw_smpl_asf()
  # draw_joints_in_motion_wrapper()
  # smpl_joints_test()
  # test()
  draw_nopose()



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
