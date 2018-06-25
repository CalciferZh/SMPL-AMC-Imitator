from smpl_np import SMPLModel
from skeleton import Joint
from skeleton import SMPLJoints
from skeleton import asf_smpl_map
from skeleton import smpl_asf_map
import numpy as np
import transforms3d

class Imitator:
  def __init__(self, asf_joints, smpl):
    asf_joints['root'].reset_pose()
    self.smpl = smpl
    self.asf_joints = asf_joints
    self.smpl_joints = self.setup_smpl_joints()
    self.align_smpl_asf()

  def setup_smpl_joints(self):
    joints = {}
    for i in range(24):
      joints[i] = SMPLJoints(i)
    for child, parent in self.smpl.parent.items():
      joints[child].parent = joints[parent]
      joints[parent].children.append(joints[child])
    J = np.copy(self.smpl.J)
    for j in joints.values():
      j.coordinate = J[j.idx]
    for j in joints.values():
      j.init_bone()
    return joints

  def align_smpl_asf(self):
    '''Return a R to align default smpl and asf to the same pose.
    Process legs only (femur and tibia)'''

    for bone_name in ['lfemur', 'rfemur']:
      asf_dir = self.asf_joints[bone_name].direction

      smpl_leg_root = self.smpl_joints[asf_smpl_map[bone_name]]
      # leg rotation -- not good
      # if bone_name == 'lfemur':
      #   smpl_leg_root.align_R = transforms3d.euler.axangle2mat([0, 1, 0], -np.pi/16)
      # else:
      #   smpl_leg_root.align_R = transforms3d.euler.axangle2mat([0, 1, 0], +np.pi/16)

      smpl_knee = smpl_leg_root.children[0]
      smpl_dir = smpl_knee.to_parent / np.linalg.norm(smpl_knee.to_parent)

      smpl_leg_root.align_R = smpl_leg_root.align_R.dot(self.compute_rodrigues(smpl_dir, asf_dir))

    for bone_name in ['ltibia', 'rtibia']:
      asf_tibia_dir = self.asf_joints[bone_name].direction
      asf_femur_dir = self.asf_joints[bone_name].parent.direction
      if not np.allclose(asf_femur_dir, asf_tibia_dir):
        # this case shouldn't happend in CMU dataset
        # so we just leave it here
        print('error: femur and tibia are different!')
        exit()

      smpl_knee = self.smpl_joints[asf_smpl_map[bone_name]]
      smpl_ankle = smpl_knee.children[0]
      smpl_tibia_dir = smpl_ankle.to_parent
      smpl_femur_dir = smpl_knee.to_parent

      smpl_knee.align_R = smpl_knee.parent.align_R.dot(self.compute_rodrigues(smpl_tibia_dir, smpl_femur_dir))

  def compute_rodrigues(self, x, y):
    ''' y = Rx '''
    theta = np.arccos(np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    axis = np.squeeze(np.cross(x, y))
    return transforms3d.axangles.axangle2mat(axis, theta)

  def map_R_asf_smpl(self):
    R = {}
    for k, v in smpl_asf_map.items():
      R[k] = self.asf_joints[v].relative_R
    return R, np.copy(np.squeeze(self.asf_joints['root'].coordinate))

  def smpl_joints_to_mesh(self):
    G = np.empty([len(self.smpl_joints), 4, 4])
    for j in self.smpl_joints.values():
      G[j.idx] = j.export_G()
    self.smpl.do_skinning(G)

  def asf_to_smpl_joints(self):
    R, offset = self.map_R_asf_smpl()
    # self.smpl_joints[0].coordinate = offset
    self.smpl_joints[0].set_motion_R(R)
    self.smpl_joints[0].update_coord()

  def set_asf_motion(self, motion):
    self.asf_joints['root'].set_motion(motion)
    self.asf_to_smpl_joints()
    self.smpl_joints_to_mesh()

  def imitate(self, motion):
    self.set_asf_motion(motion)


if __name__ == '__main__':
  import reader
  subject = '01'
  im = Imitator(
    reader.parse_asf('./data/%s/%s.asf' % (subject, subject)),
    SMPLModel('./model.pkl')
  )

  sequence = '01'
  frame_idx = 0
  motions = reader.parse_amc('./data/%s/%s_%s.amc' % (subject, subject, sequence))
  im.imitate(motions[frame_idx])
  im.smpl.output_mesh('./posed.obj')

