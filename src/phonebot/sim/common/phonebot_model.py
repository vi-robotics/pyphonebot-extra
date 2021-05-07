#!/usr/bin/env python3
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.transform import Transform, Rotation, Position
from phonebot.sim.common.model import *


class PhonebotModel(object):
    """
    Canonical phonebot definition via abstraction.

    """

    def __init__(self, config=PhonebotSettings()):
        self.config_ = config
        self.links_ = []
        self.joints_ = []

    def append_leg_half(self, prefix, suffix):
        config = self.config_
        leg_frame = '{}_leg_origin'.format(prefix)
        hip_frame = '{}_hip_link_{}'.format(prefix, suffix)
        knee_frame = '{}_knee_link_{}'.format(prefix, suffix)
        foot_frame = '{}_foot_link_{}'.format(prefix, suffix)

        hip_rotation = Rotation.from_axis_angle(
            [0, 1, 0, config.hip_angle[suffix]])
        hip_joint_pose = Transform(
            Position([config.hip_sign[suffix] * config.hip_joint_offset, 0, 0]), hip_rotation)
        hip_pose = hip_joint_pose * \
            Transform.from_position([config.hip_link_length / 2.0, 0.0, 0.0])
        hip_geom = ModelCylinderGeometry('hip', [
            config.leg_radius, config.hip_link_length])
        hip_mass = config.hip_mass
        hip_link = ModelLink(name=hip_frame, pose=ModelPose(hip_pose, leg_frame), mass=hip_mass,
                             inertia=ModelInertia.from_geometry(hip_mass, hip_geom), geometry=hip_geom)
        hip_joint = ModelJoint(name='{}_hip_joint_{}'.format(prefix, suffix), type=ModelJoint.REVOLUTE, parent=leg_frame, child=hip_frame, pose=ModelPose(pose=hip_joint_pose, frame=leg_frame), axis=[
            0, 0, 1])
        self.links_.append(hip_link)
        self.joints_.append(hip_joint)
        knee_joint_pose = Transform(Position(
            [config.hip_link_length / 2.0, 0, 0]), Rotation.from_axis_angle([0, 0, 1, np.pi]))
        knee_pose = knee_joint_pose * \
            Transform.from_position([config.knee_link_length / 2.0, 0, 0])
        knee_geom = ModelCylinderGeometry('knee', [
            config.leg_radius, config.knee_link_length])
        knee_mass = config.knee_mass
        knee_link = ModelLink(name=knee_frame, pose=ModelPose(knee_pose, hip_frame), mass=knee_mass,
                              inertia=ModelInertia.from_geometry(knee_mass, knee_geom), geometry=knee_geom)
        knee_joint = ModelJoint(name='{}_knee_joint_{}'.format(prefix, suffix), type=ModelJoint.REVOLUTE, parent=hip_frame, child=knee_frame, pose=ModelPose(pose=knee_joint_pose, frame=hip_frame), axis=[
            0, 0, 1])
        self.links_.append(knee_link)
        self.joints_.append(knee_joint)
        foot_mass = config.foot_mass
        foot_geom = ModelSphereGeometry('foot', config.leg_radius)
        foot_pose = ModelPose(Transform.from_position([
            config.knee_link_length / 2.0, 0, 0]), knee_frame)
        foot_link = ModelLink(name=foot_frame, pose=foot_pose, mass=foot_mass,
                              inertia=ModelInertia.from_geometry(foot_mass, foot_geom), geometry=foot_geom)
        foot_joint = ModelJoint(name='{}_foot_joint_{}'.format(prefix, suffix), type=ModelJoint.FIXED, parent=knee_frame, child=foot_frame, pose=foot_pose, axis=[
            0, 0, 1])
        self.links_.append(foot_link)
        self.joints_.append(foot_joint)

    def append_leg(self, prefix):
        config = self.config_
        leg_index = config.index[prefix]
        leg_frame = '{}_leg_origin'.format(prefix)
        leg_sign = config.leg_sign[leg_index]
        leg_pose = Transform(np.multiply(
            leg_sign, config.leg_offset), config.leg_rotation[leg_index].to_quaternion())
        geom = ModelSphereGeometry('leg_o', config.leg_radius)
        fake_mass = 1e-6
        link = ModelLink(name=leg_frame, pose=ModelPose(leg_pose, 'body'), mass=fake_mass,
                         inertia=ModelInertia.from_geometry(fake_mass, geom), geometry=geom)
        joint = ModelJoint(name='{}_leg_origin_joint'.format(prefix), type=ModelJoint.FIXED, parent='body', child=leg_frame, pose=ModelPose(leg_pose, frame='body'), axis=[
            0, 0, 1])
        self.links_.append(link)
        self.joints_.append(joint)
        for suffix in 'ab':
            self.append_leg_half(prefix, suffix)

    def append_root(self):
        config = self.config_
        body_geom = ModelBoxGeometry('body', config.body_dim)
        body_mass = config.body_mass
        body = ModelLink(name='body', pose=ModelPose(Transform.identity(), 'body'), mass=body_mass,
                         inertia=ModelInertia.from_geometry(body_mass, body_geom), geometry=body_geom)
        self.links_.append(body)

    def build(self):
        self.append_root()
        for prefix in self.config_.order:
            self.append_leg(prefix)

        return (self.joints_, self.links_)
