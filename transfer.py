from skeleton import *
from mathtool import *


def align_smpl_asf(asf_joints, smpl_joints):
  '''Return a R to align default smpl and asf to the same pose.
  Process legs only (femur and tibia)'''
  R = np.stack([np.eye(3) for k in range(24)], axis=0)

  for bone_name in ['lfemur', 'rfemur']:
    asf_dir = asf_joints[bone_name].direction

    smpl_root = smpl_joints[asf_smpl_map[bone_name]]
    smpl_knee = smpl_root.children[0]
    smpl_dir = smpl_knee.coordinate - smpl_root.coordinate

    R[smpl_root.idx] = compute_rodrigues(smpl_dir, asf_dir)

  return R



