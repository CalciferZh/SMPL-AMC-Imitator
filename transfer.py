from skeleton import asf_smpl_map
from mathtool import *


def align_smpl_asf(asf_joints, smpl_joints):
  '''Return a R to align default smpl and asf to the same pose.
  Process legs only (femur and tibia)'''
  R = np.stack([np.eye(3) for k in range(24)], axis=0)

  for bone_name in ['lfemur', 'rfemur']:
    asf_dir = asf_joints[bone_name].direction

    smpl_root = smpl_joints[asf_smpl_map[bone_name]]
    smpl_knee = smpl_root.children[0]
    smpl_dir = smpl_knee.to_parent

    R[smpl_root.idx] = compute_rodrigues(smpl_dir, asf_dir)

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

    R[smpl_knee.idx] = compute_rodrigues(smpl_tibia_dir, smpl_femur_dir)

  return R




