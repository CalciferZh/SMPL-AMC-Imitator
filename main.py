import reader
from smpl_np import SMPLModel
from skeleton import *
from vistool import *
from transfer import *
from 3Dviewer.py import *


def draw_pose_static():
  asf_joints = reader.parse_asf('./data/01/01.asf')
  asf_joints['root'].reset_pose()

  smpl = SMPLModel('./model.pkl')
  smpl_joints = setup_smpl_joints(smpl)

  align_smpl_asf(asf_joints, smpl_joints)

  frame_idx = 180
  motion = reader.parse_amc('./data/01/01_01.amc')[frame_idx]
  asf_joints['root'].set_motion(motion)

  R, offset = map_R_asf_smpl(asf_joints)
  smpl_joints[0].set_motion_R(R)
  smpl_joints[0].update_coord()

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
  draw_pose_static()
