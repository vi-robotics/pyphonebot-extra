# NOTE(ycho): this locks python version to >=3.7
# from __future__ import annotations 
from abc import abstractmethod
from enum import IntEnum
from numbers import Number
from typing import Any, Tuple, Union
import numpy as np
from phonebot.core.common.math.transform import Transform, Position, Rotation


class ModelPose():
    """A ModelPose contains the frame and pose relative to the provided
    frame.
    """

    def __init__(
            self, pose: Transform = Transform.identity(),
            frame: str = None):

        self.pose = pose
        self.frame = frame

    @property
    def position(self) -> Position:
        """The relative position in the given frame

        Returns:
            Position: The resulting Position object in the given frame
        """
        return self.pose.position

    @property
    def rotation(self) -> Rotation:
        """The relative rotation in the given frame

        Returns:
            Rotation: The resulting Rotation object in the given frame
        """
        return self.pose.rotation

    def __str__(self):
        return '({}[{}])'.format(self.pose.__str__(), self.frame)

    def __repr__(self):
        return '({}[{}])'.format(self.pose.__repr__(), self.frame)


class ModelGeometry():
    """A base class for a model which has a name and inertia calculation
    """

    def __init__(self, name: str):
        self.name = name

    @property
    def name(self) -> str:
        """The name of the geometry.

        Returns:
            str: The name of the geometry.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the name of the geometry.

        Args:
            value (str): A string to set the name to.

        Raises:
            ValueError: if value is not a string
        """
        if not isinstance(value, str):
            raise ValueError(f"value must be a str, not {type(value)}")
        else:
            self._name = value

    @abstractmethod
    def inertia(self) -> np.ndarray:
        """Returns an inertia 3x3 matrix.

        Returns:
            np.ndarray: A 3x3 rotational inertia matrix.
        """
        return NotImplemented


class ModelParametricGeometry(ModelGeometry):
    """A ModelGeometry with an additional param attribute which can contain
    parameters, as well as a type.
    """

    def __init__(self, name: str, type: str, param: Union[Any, Tuple[Any]]):
        """Initializes a ModelParametricGeometry object which inherits from
        ModelGeometry and contains an additional param tuple as well as a
        string representing the type.

        Args:
            name (str): The name of the geometry
            type (str): The type of the geometry
            param (Union[Any, Tuple[Any]]): A parameter or tuple of parameters
                defining the geometry.
        """
        super().__init__(name)
        self.type = type
        self.param = param

    @property
    def type(self) -> str:
        """The type of the geometry

        Returns:
            str: The type of the geometry
        """
        return self._type

    @type.setter
    def type(self, value: str):
        """Set the type string of the geometry

        Args:
            value (str): The geometry type
        """
        if not isinstance(value, str):
            raise ValueError(f"value must be type str, not {type(value)}")
        self._type = value

    @property
    def param(self) -> Tuple[Any]:
        """The parameter set

        Returns:
            Tuple[Any]: The tuple of parameters for the geometry
        """
        return self._param

    @param.setter
    def param(self, value: Tuple[Any]):
        """Set the parameters of the geometry

        Args:
            value (Tuple[Any]): The parameters to set
        """
        self._param = value

    def __eq__(self, other):
        if not self.type == other.type:
            return False
        return self.param == other.param


class ModelBoxGeometry(ModelParametricGeometry):
    """A parametric Box geometry defined from length, width, and height
    dimensions
    """

    def __init__(self, name: str, param: Tuple[Number]):
        """Initialize a ModelBoxGeometry parametrically.

        Args:
            name (str): The name of the geometry.
            param (Tuple[Any]): A 3-tuple of x, y, z dimensions.
        """
        super().__init__(name, 'box', param)

    def inertia(self) -> np.ndarray:
        """The inertia matrix of the box, assuming that the box is made of
        a uniform density material.

        Returns:
            np.ndarray: A 3x3 inertia array
        """
        dims = self.dimensions
        dim_sq = np.square(dims)
        return np.diag(dim_sq.sum() - dim_sq) / 12

    @property
    def dimensions(self) -> Tuple[Number]:
        """The length, width and height of the box.

        Returns:
            Tuple[Number]: A Tuple of length, width, and height dimensions (x,y,
                z)
        """
        return self.param

    @property
    def x(self) -> Number:
        """The x dimension

        Returns:
            Number: The length of the box
        """
        return self.param[0]

    @property
    def y(self) -> Number:
        """The y dimension

        Returns:
            Number: The width of the box
        """
        return self.param[1]

    @property
    def z(self) -> Number:
        """The z dimension

        Returns:
            Number: The height of the box
        """
        return self.param[2]


class ModelCylinderGeometry(ModelParametricGeometry):
    """A parametric Cylinder geometry defined from radius and length
    dimensions
    """

    def __init__(self, name: str, param: Tuple[Number]):
        """Initializes a ModelCylinderGeometry parametrically

        Args:
            name (str): The name of the geometry
            param (Tuple[Any]): A 2-tuple of radius and length
        """
        super().__init__(name, 'cylinder', param)

    def inertia(self) -> np.ndarray:
        """The inertia matrix of the cylinder, assuming that the cylinder is
        made of a uniform density material.

        Returns:
            np.ndarray: A 3x3 inertia array
        """
        r, h = self.radius, self.height
        ixx = iyy = (r**2 + h**2) / 12
        izz = 0.5 * r**2
        return np.diag([ixx, iyy, izz])

    @property
    def radius(self) -> Number:
        """The radius of the cylinder

        Returns:
            Number: The radius of the cylinder
        """
        return self.param[0]

    @property
    def height(self) -> Number:
        """The height of the cylinder. This is the same as the length.

        Returns:
            Number: The height of the cylinder
        """
        return self.param[1]

    @property
    def length(self) -> Number:
        """The length of the cylinder. This is the same as the height.

        Returns:
            Number: The length of the cylinder
        """
        return self.param[1]


class ModelSphereGeometry(ModelParametricGeometry):
    """A parametric Sphere geometry defined from radius
    """

    def __init__(self, name: str, param: Number):
        """Initializes a ModelSphereGeometry parametrically

        Args:
            name (str): The name of the geometry
            param (Number): The radius of the sphere.
        """
        super().__init__(name, 'sphere', param)

    def inertia(self) -> np.ndarray:
        """The inertia matrix of the sphere, assuming that the sphere is
        made of a uniform density material.

        Returns:
            np.ndarray: A 3x3 inertia array
        """
        r = self.radius
        return 0.4 * r**2 * np.eye(3)

    @property
    def radius(self) -> Number:
        """The radius of the sphere

        Returns:
            Number: The radius of the sphere
        """
        return self.param


class ModelInertia():
    """A model's inertia with an optional relative pose.
    """

    def __init__(self, inertia: np.ndarray, pose: ModelPose = None):
        """Initialize a ModelInerita object

        Args:
            inertia (np.ndarray): A 3x3 inertia matrix
            pose (ModelPose, optional): A relative transform for the model.
                Defaults to None.
        """
        if pose is None:
            pose = ModelPose()
        self.inertia = inertia
        self.pose = pose

    @classmethod
    def from_geometry(cls, mass: Number, geom: ModelGeometry) -> 'ModelInertia':
        """Create a ModelInertia instance using the inertia from an existing
        geometry.

        Args:
            mass (Number): The mass of the object
            geom (ModelGeometry): A ModelGeometry object to create the inertia
                for.

        Returns:
            ModelInertia: The resulting model inertia object
        """
        imat = mass * geom.inertia()
        return cls.from_matrix(imat)

    @classmethod
    def from_matrix(cls, imat: np.ndarray) -> 'ModelInertia':
        """Create a ModelInertia instance from an inertia matrix

        Args:
            imat (np.ndarray): A 3x3 inerita matrix

        Returns:
            ModelInertia: The resulting ModelInertia object
        """
        return cls(imat)


class ModelLink(object):
    """A link of a model
    """

    def __init__(self, name: str, pose: ModelPose, mass: float,
                 inertia: ModelInertia, geometry: ModelGeometry):
        """Initialize a link with relavent physical properties

        Args:
            name (str): The name of the ModelLink object
            pose (ModelPose): The relative ModelPose of the link
            mass (float): The mass of the link
            inertia (ModelInertia): The inertia of the link
            geometry (ModelGeometry): The geometry of the link
        """

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


class ModelJointType(IntEnum):
    """The type of model joint
    """
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2


class ModelJoint():
    """A joint of a model
    """

    def __init__(self, name: str, type: ModelJointType, parent: str,
                 child: str, pose: ModelPose, axis: np.ndarray):
        """Initialize a joint with relevant physical properties

        Args:
            name (str): The name of the ModelJoint object
            type (ModelJointType): The type of joint
            parent (str): The name of the parent of the joint
            child (str): The name of the child of the joint
            pose (ModelPose): The transform of the joint location
            axis (np.ndarray): A 1x3 array representing the joint axis.
        """
        self.name = name
        self.type = type
        self.parent = parent
        self.child = child
        self.pose = pose
        self.axis = axis

    def __str__(self):
        return '[{}] : {} ({}->{}), {}, {}'.format(self.name,
                                                   self.type, self.parent,
                                                   self.child, self.pose, self.axis)

    def __repr__(self):
        return self.__str__()
