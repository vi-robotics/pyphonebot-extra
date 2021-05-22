#!/usr/bin/env python3
from typing import List
from numpy import isin
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.transform import Transform, Rotation, Position
from pyphonebot_extra.sim.common.model import *


class PhonebotModel():
    """
    Canonical phonebot definition via abstraction.
    """

    def __init__(self, config: PhonebotSettings = None):
        """Create a PhonebotModel object using a given PhonebotSettings config
        object.

        Args:
            config (PhonebotSettings, optional): The config object to define the
                joint and link positions for the legs. Defaults to None.
        """

        if config is None:
            config = PhonebotSettings()
        self.config = config
        self.links = []
        self.joints = []

    @property
    def joints(self) -> List[ModelJoint]:
        """The PhonebotModel list of joints

        Returns:
            List[ModelJoint]: The current list of ModelJoint objects
        """
        return self.joints_

    @joints.setter
    def joints(self, value: List[ModelJoint]):
        """Set the PhonebotModel list of joints

        Args:
            value (List[ModelJoint]): A list of ModelJoint objects to set

        Raises:
            ValueError: if value is not a list of ModelJoint objects
        """
        if not all([isinstance(el, ModelJoint) for el in value]):
            raise ValueError("value must be a list of ModelJoint objects")
        else:
            self.joints_ = value

    @property
    def links(self) -> List[ModelLink]:
        """The PhonebotModel list of links

        Returns:
            List[ModelLink]: The current list of ModelLink objects
        """
        return self.links_

    @links.setter
    def links(self, value: List[ModelLink]):
        """Set the PhonebotModel list of links

        Args:
            value (List[ModelLink]): A list of ModelLink objects to set

        Raises:
            ValueError: if value is not a list of ModelLink objects
        """

        if not all([isinstance(el, ModelLink) for el in value]):
            raise ValueError("value must be a list of ModelLink objects")
        else:
            self.links_ = value

    @property
    def config(self) -> PhonebotSettings:
        """The PhoneBot configuration settings which define the joint locations

        Returns:
            PhonebotSettings: The current PhonebotSettings object
        """
        return self.config_

    @config.setter
    def config(self, value: PhonebotSettings):
        """Sets the joint configuration.

        Args:
            value (PhonebotSettings): The PhonebotSettings object to set

        Raises:
            ValueError: if value is not a PhonebotSettings object.
        """
        if not isinstance(value, PhonebotSettings):
            raise ValueError("config must be a PhoneBotSettings type,"
                             f" not {type(value)}")
        else:
            self.config_ = value

    def append_leg_half(self, prefix: str, suffix: str):
        """Construct half of a leg who's joints and links are named with the
        provided prefix and suffix. The suffix must be a key in the
        config.hip_sign dictionary so the proper joint signs can be assigned.

        Args:
            prefix (str): The prefix to use for naming the joints and links.
            suffix (str): The suffix to use for naming the joints and links, and
                additionally setting the sign of the leg half.
        """

        # Set up the frame names
        config = self.config
        leg_frame = '{}_leg_origin'.format(prefix)
        hip_frame = '{}_hip_link_{}'.format(prefix, suffix)
        knee_frame = '{}_knee_link_{}'.format(prefix, suffix)
        foot_frame = '{}_foot_link_{}'.format(prefix, suffix)

        # Create the hip joint and link
        # The direction of rotation for the hip will be set with the suffix from
        # looking up the suffix in config.hip_ange
        hip_rotation = Rotation.from_axis_angle(
            [0, 1, 0, config.hip_angle[suffix]])
        hip_joint_pose = Transform(
            Position([config.hip_sign[suffix] * config.hip_joint_offset, 0, 0]),
            hip_rotation)
        hip_pose = hip_joint_pose * \
            Transform.from_position([config.hip_link_length / 2.0, 0.0, 0.0])
        hip_geom = ModelCylinderGeometry('hip', [
            config.leg_radius, config.hip_link_length])
        hip_mass = config.hip_mass
        hip_link = ModelLink(name=hip_frame, pose=ModelPose(hip_pose, leg_frame),
                             mass=hip_mass,
                             inertia=ModelInertia.from_geometry(hip_mass, hip_geom),
                             geometry=hip_geom)
        hip_joint = ModelJoint(
            name='{}_hip_joint_{}'.format(prefix, suffix),
            type=ModelJointType.REVOLUTE, parent=leg_frame, child=hip_frame,
            pose=ModelPose(pose=hip_joint_pose, frame=leg_frame),
            axis=np.asarray([0, 0, 1]))
        self.links.append(hip_link)
        self.joints.append(hip_joint)

        # Create the knee joint and link
        knee_joint_pose = Transform(Position(
            [config.hip_link_length / 2.0, 0, 0]),
            Rotation.from_axis_angle([0, 0, 1, np.pi]))
        knee_pose = knee_joint_pose * \
            Transform.from_position([config.knee_link_length / 2.0, 0, 0])
        knee_geom = ModelCylinderGeometry('knee', [
            config.leg_radius, config.knee_link_length])
        knee_mass = config.knee_mass
        knee_link = ModelLink(name=knee_frame,
                              pose=ModelPose(knee_pose, hip_frame), mass=knee_mass,
                              inertia=ModelInertia.from_geometry(knee_mass, knee_geom),
                              geometry=knee_geom)
        knee_joint = ModelJoint(
            name='{}_knee_joint_{}'.format(prefix, suffix),
            type=ModelJointType.REVOLUTE, parent=hip_frame, child=knee_frame,
            pose=ModelPose(pose=knee_joint_pose, frame=hip_frame),
            axis=np.asarray([0, 0, 1]))
        self.links.append(knee_link)
        self.joints.append(knee_joint)

        # Create the foot joint and link
        foot_mass = config.foot_mass
        foot_geom = ModelSphereGeometry('foot', config.leg_radius)
        foot_pose = ModelPose(Transform.from_position([
            config.knee_link_length / 2.0, 0, 0]), knee_frame)
        foot_link = ModelLink(name=foot_frame, pose=foot_pose, mass=foot_mass,
                              inertia=ModelInertia.from_geometry(foot_mass, foot_geom),
                              geometry=foot_geom)
        foot_joint = ModelJoint(
            name='{}_foot_joint_{}'.format(prefix, suffix),
            type=ModelJointType.FIXED, parent=knee_frame, child=foot_frame,
            pose=foot_pose, axis=np.asarray([0, 0, 1]))
        self.links.append(foot_link)
        self.joints.append(foot_joint)

    def append_leg(self, prefix: str):
        """Add a full leg to the PhonebotModel using the provided prefix to name
        the joints and links. Each leg half will have suffixes "a" and "b"
        which refer to each half of the leg.

        Args:
            prefix (str): The leg prefix for naming joints and links. It is also
                used to look up the leg in the config.index dictionary, and the
                leg parameters will be taken from the appropriate leg.
        """
        config = self.config
        leg_index = config.index[prefix]
        leg_frame = '{}_leg_origin'.format(prefix)
        leg_sign = config.leg_sign[leg_index]
        leg_pose = Transform(np.multiply(
            leg_sign, config.leg_offset),
            config.leg_rotation[leg_index].to_quaternion())
        geom = ModelSphereGeometry('leg_o', config.leg_radius)
        fake_mass = 1e-6
        link = ModelLink(name=leg_frame, pose=ModelPose(leg_pose, 'body'),
                         mass=fake_mass,
                         inertia=ModelInertia.from_geometry(fake_mass, geom),
                         geometry=geom)
        joint = ModelJoint(name='{}_leg_origin_joint'.format(prefix),
                           type=ModelJointType.FIXED,
                           parent='body', child=leg_frame,
                           pose=ModelPose(leg_pose, frame='body'),
                           axis=np.asarray([0, 0, 1]))
        self.links.append(link)
        self.joints.append(joint)
        for suffix in 'ab':
            self.append_leg_half(prefix, suffix)

    def append_root(self):
        """Add the main body to the PhonebotModel
        """
        config = self.config
        body_geom = ModelBoxGeometry('body', config.body_dim)
        body_mass = config.body_mass
        body = ModelLink(name='body',
                         pose=ModelPose(Transform.identity(), 'body'),
                         mass=body_mass,
                         inertia=ModelInertia.from_geometry(body_mass, body_geom),
                         geometry=body_geom)
        self.links.append(body)

    def build(self):
        """Build the PhonebotModel by creating the main body and the legs using
        the config.

        Returns:
            Tuple[List[ModelJoint], List[ModelLink]]: A tuple comprising:
                List[ModelJoint]: The joints of the Phonebot
                List[ModelLink]: The links of the Phonebot
        """
        self.append_root()
        for prefix in self.config.order:
            self.append_leg(prefix)

        return (self.joints, self.links)
