import numpy as np


class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    self.name = name
    self.direction = np.matrix(direction)
    self.length = length
    self.C, self.Cinv = rotation_matrix_axis(axis)
    self.limits = np.zeros([3, 2])
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
      motion['root'] = motion['root'][3:]
      self.matrix = rotation_matrix(self.C, self.Cinv, motion[self.name])
    else:
      # set rx ry rz according to degree of freedom
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      self.matrix = rotation_matrix(self.C, self.Cinv, rotation) * self.parent.matrix
      self.coordinate = np.squeeze(np.array(self.parent.coordinate + self.length * self.direction * self.matrix))
    for child in self.children:
      child.set_motion(motion)

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def rotation_matrix_axis(axis):
    # Change coordinate system through matrix C
  rx = np.deg2rad(float(axis[0]))
  ry = np.deg2rad(float(axis[1]))
  rz = np.deg2rad(float(axis[2]))

  Cx = np.matrix([[1, 0, 0],
                  [0, np.cos(rx), np.sin(rx)],
                  [0, -np.sin(rx), np.cos(rx)]])

  Cy = np.matrix([[np.cos(ry), 0, -np.sin(ry)],
                  [0, 1, 0],
                  [np.sin(ry), 0, np.cos(ry)]])

  Cz = np.matrix([[np.cos(rz), np.sin(rz), 0],
                  [-np.sin(rz), np.cos(rz), 0],
                  [0, 0, 1]])

  C = Cx * Cy * Cz
  Cinv = np.linalg.inv(C)
  return C, Cinv


def rotation_matrix(C, Cinv, motion):
    # Construct rotation matrix M
    tx = np.deg2rad(motion[0])
    ty = np.deg2rad(motion[1])
    tz = np.deg2rad(motion[2])

    Mx = np.matrix([[1, 0, 0],
                    [0, np.cos(tx), np.sin(tx)],
                    [0, -np.sin(tx), np.cos(tx)]])

    My = np.matrix([[np.cos(ty), 0, -np.sin(ty)],
                    [0, 1, 0],
                    [np.sin(ty), 0, np.cos(ty)]])

    Mz = np.matrix([[np.cos(tz), np.sin(tz), 0],
                    [-np.sin(tz), np.cos(tz), 0],
                    [0, 0, 1]])
    M = Mx * My * Mz
    L = Cinv * M * C
    return L


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

