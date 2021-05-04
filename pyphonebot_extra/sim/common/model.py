# uncompyle6 version 3.5.1
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Oct  8 2019, 13:06:37)
# [GCC 5.4.0 20160609]
# Embedded file name: /home/jamiecho/Repos/PhoneBot/control/sim/common/model.py
# Compiled at: 2019-11-19 21:36:28
# Size of source mod 2**32: 4059 bytes
from abc import abstractmethod
import numpy as np
from phonebot.core.common.math.transform import Transform, Position, Rotation


class ModelPose(object):

    def __init__(
            self, pose: Transform = Transform.identity(),
            frame: str = None):
        self.pose = pose
        self.frame = frame

    @property
    def position(self):
        return self.pose.position

    @property
    def rotation(self):
        return self.pose.rotation

    def __str__(self):
        return '({}[{}])'.format(self.pose.__str__(), self.frame)

    def __repr__(self):
        return '({}[{}])'.format(self.pose.__repr__(), self.frame)


class ModelGeometry(object):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def inertia(self):
        return NotImplemented


class ModelParametricGeometry(ModelGeometry):

    def __init__(self, name, type, param):
        super().__init__(name)
        self.type_ = type
        self.param_ = param

    @property
    def type(self):
        return self.type_

    @property
    def param(self):
        return self.param_

    def __eq__(self, other):
        if not self.type == other.type:
            return False
        return self.param == other.param


class ModelBoxGeometry(ModelParametricGeometry):

    def __init__(self, name, param):
        super().__init__(name, 'box', param)

    def inertia(self):
        dims = self.dimensions
        dim_sq = np.square(dims)
        return 0.08333333333333333 * np.diag(dim_sq.sum() - dim_sq)

    @property
    def dimensions(self):
        return self.param_

    @property
    def x(self):
        return self.param_[0]

    @property
    def y(self):
        return self.param_[1]

    @property
    def z(self):
        return self.param_[2]


class ModelCylinderGeometry(ModelParametricGeometry):

    def __init__(self, name, param):
        super().__init__(name, 'cylinder', param)

    def inertia(self):
        r, h = self.radius, self.height
        ixx = iyy = 0.08333333333333333 * (r * r + h * h)
        izz = 0.5 * r * r
        return np.diag([ixx, iyy, izz])

    @property
    def radius(self):
        return self.param_[0]

    @property
    def height(self):
        return self.param_[1]

    @property
    def length(self):
        return self.param_[1]


class ModelSphereGeometry(ModelParametricGeometry):

    def __init__(self, name, param):
        super().__init__(name, 'sphere', param)

    def inertia(self):
        r = self.radius
        return 0.4 * r * r * np.eye(3)

    @property
    def radius(self):
        return self.param_


class ModelInertia(object):

    def __init__(self, inertia, pose: ModelPose = None):
        if pose is None:
            pose = ModelPose()
        self.inertia = inertia
        self.pose = pose

    @classmethod
    def from_geometry(cls, mass, geom: ModelGeometry):
        imat = mass * geom.inertia()
        return cls.from_matrix(imat)

    @classmethod
    def from_matrix(cls, imat):
        return cls(imat)


class ModelLink(object):

    def __init__(self, name: str, pose: ModelPose, mass: float,
                 inertia: ModelInertia, geometry: ModelGeometry):
        self.name = name
        self.pose = pose
        self.mass = mass
        self.inertia = inertia
        self.geometry = geometry

    def __str__(self):
        return '{}'.format(self.name)

    def __repr__(self):
        return '[{}] : {}, {}, {}, {}'.format(
            self.name, self.pose, self.mass, self.inertia, self.geometry)


class ModelJoint(object):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2

    def __init__(self, name: str, type: int, parent: str,
                 child: str, pose: ModelPose, axis):
        self.name = name
        self.type = type
        self.parent = parent
        self.child = child
        self.pose = pose
        self.axis = axis

    def __str__(self):
        return '[{}] : {} ({}->{}), {}, {}'.format(self.name,
                                                   self.type, self.parent, self.child, self.pose, self.axis)

    def __repr__(self):
        return self.__str__()
# okay decompiling model.cpython-35.pyc
