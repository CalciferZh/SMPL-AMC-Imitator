import numpy as np
import transforms3d


class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    self.name = name
    self.direction = np.matrix(direction)
    self.length = length
    axis = np.deg2rad(axis)
    self.C = np.matrix(transforms3d.euler.euler2mat(*axis))
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    self.movable = len(dof) == 0
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.array(motion['root'][:3])
      self.coordinate = np.zeros(3)
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C * np.matrix(transforms3d.euler.euler2mat(*rotation)) * self.Cinv
    else:
      # set rx ry rz according to degree of freedom
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix * self.C * np.matrix(transforms3d.euler.euler2mat(*rotation)) * self.Cinv
      self.coordinate = np.squeeze(np.array(np.reshape(self.parent.coordinate, [3, 1]) + self.length * self.matrix * np.reshape(self.direction, [3, 1])))
    for child in self.children:
      child.set_motion(motion)

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':bonedata':
      content = content[idx+1:]

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'
    # use int for approximation
    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])  # skip '('
        maxi = float(line[1][:-1])  # skip ')'
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'
  line, idx = read_line(content, idx)
  assert line[0] == 'begin'
  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]
  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric()
  while True:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        return frames
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)


def joint_semantic():
  # only used for body-drawing
  ret = {
    0: 'root',
    1: 'lhipjoint',
    2: 'rhipjoint',
    3: 'lowerback',
    4: 'lfemur',
    5: 'rfemur',
    6: 'upperback',
    7: 'ltibia',
    8: 'rtibia',
    9: 'thorax',
    10: 'lfoot',
    11: 'rfoot',
    12: 'lowerneck',
    13: 'lclavicle',
    14: 'rclavicle',
    15: 'upperneck',
    16: 'lhumerus',
    17: 'rhumerus',
    18: 'lradius',
    19: 'rradius',
    20: 'lwrist',
    21: 'rwrist',
    22: 'lhand',
    23: 'rhand'
  }
  return ret


def smpl_asf_map():
  ret = {
    0: 'root',
    1: 'lfemur',
    2: 'rfemur',
    3: 'upperback',
    4: 'ltibia',
    5: 'rtibia',
    6: 'thorax',
    7: 'lfoot',
    8: 'rfoot',
    9: 'lowerneck',
    10: 'ltoes',
    11: 'rtoes',
    12: 'upperneck',
    13: 'lclavicle',
    14: 'rclavicle',
    15: 'head',
    16: 'lhumerus',
    17: 'rhumerus',
    18: 'lradius',
    19: 'rradius',
    20: 'lwrist',
    21: 'rwrist',
    22: 'lhand',
    23: 'rhand'
  }
  return ret


def asf_smpl_map():
  sa_map = smpl_asf_map()
  index = {}
  for k, v in sa_map.items():
    index[v] = k
  return index


def joint_index():
  semantic = joint_semantic()
  index = {}
  for k, v in semantic.items():
    index[v] = k
  return index


class SMPLJoints:
  def __init__(self, idx):
    self.idx = idx
    self.to_parent = None
    self.parent = None
    self.coordinate = None
    self.matrix = None
    self.children = []

  def update_info(self):
    if self.parent is not None:
      self.to_parent = self.coordinate - self.parent.coordinate

  def set_motion(self, R):
    if self.parent is not None:
      self.coordinate = self.parent.coordinate + np.squeeze(np.dot(self.parent.matrix, np.reshape(self.to_parent, [3,1])))
      self.matrix = np.dot(self.parent.matrix, R[self.idx])
    else:
      self.matrix = R[self.idx]
    for child in self.children:
      child.set_motion(R)

  def to_dict(self):
    ret = {self.idx: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret
