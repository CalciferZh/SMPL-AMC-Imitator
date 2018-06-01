from skeleton import asf_smpl_map
from skeleton import smpl_asf_map
from skeleton import setup_smpl_joints
from mathtool import compute_rodrigues
from smpl_np import SMPLModel
import numpy as np
import reader
import vistool



def align_smpl_asf(asf_joints, smpl_joints):
  '''Return a R to align default smpl and asf to the same pose.
  Process legs only (femur and tibia)'''
  for bone_name in ['lfemur', 'rfemur']:
    asf_dir = asf_joints[bone_name].direction

    smpl_root = smpl_joints[asf_smpl_map[bone_name]]
    smpl_knee = smpl_root.children[0]
    smpl_dir = smpl_knee.to_parent / np.linalg.norm(smpl_knee.to_parent)

    smpl_root.align_R = compute_rodrigues(smpl_dir, asf_dir)

  for bone_name in ['ltibia', 'rtibia']:
    asf_tibia_dir = asf_joints[bone_name].direction
    asf_femur_dir = asf_joints[bone_name].parent.direction
    if not np.allclose(asf_femur_dir, asf_tibia_dir):
      # this case shouldn't happend in CMU dataset
      # so we just leave it here
      print('error: femur and tibia are different!')
      exit()

    smpl_knee = smpl_joints[asf_smpl_map[bone_name]]
    smpl_root = smpl_knee.parent
    smpl_ankle = smpl_knee.children[0]
    smpl_tibia_dir = smpl_ankle.to_parent
    smpl_femur_dir = smpl_knee.to_parent

    smpl_knee.align_R = smpl_knee.parent.align_R.dot(compute_rodrigues(smpl_tibia_dir, smpl_femur_dir))


def map_R_asf_smpl(asf_joints):
  R = {}
  for k, v in smpl_asf_map.items():
    R[k] = asf_joints[v].relative_R
  return R, np.copy(np.squeeze(asf_joints['root'].coordinate))


def set_smpl(smpl_joints, smpl):
  G = np.empty([len(smpl_joints), 4, 4])
  for j in smpl_joints.values():
    G[j.idx] = j.export_G()
  smpl.do_skinning(G)


if __name__ == '__main__':
  subject = '01'
  sequence = '01'
  frame_idx = 180

  asf_joints = reader.parse_asf('./data/%s/%s.asf' % (subject, subject))
  asf_joints['root'].reset_pose()

  smpl = SMPLModel('./model.pkl')
  smpl_joints = setup_smpl_joints(smpl, False)

  align_smpl_asf(asf_joints, smpl_joints)

  motions = reader.parse_amc('./data/%s/%s_%s.amc' % (subject, subject, sequence))

  motion = motions[frame_idx]

  asf_joints['root'].set_motion(motion)
  R, offset = map_R_asf_smpl(asf_joints)
  smpl_joints[0].coordinate = offset
  smpl_joints[0].set_motion_R(R)
  smpl_joints[0].update_coord()

  set_smpl(smpl_joints, smpl)

  smpl.output_mesh('posed.obj')
