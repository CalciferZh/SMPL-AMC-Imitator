import numpy as np


class Bone:
  def __init__(self, name, direction, axis, dof, limits):
    self.name = name
    self.direction = direction
    self.axis = axis
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

  def pretty_print(self):
    print('===================================')
    print('bone: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('axis:')
    print(self.axis)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read bonedata only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':bonedata':
      content = content[idx+1:]

  # read bones
  bones = {'root': Bone('root', np.zeros(3), np.zeros(3), [], [])}
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
    bones[name] = Bone(
      name,
      direction,
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
    bones[line[0]].children = line[1:]
    for nm in line[1:]:
      bones[nm].parent = line[0]
  return bones
