#!/usr/bin/env python3

from functools import partial, lru_cache
from collections import namedtuple, defaultdict, deque
import time
import numpy as np
import pybullet as pb
from typing import List
import logging

from phonebot.core.common.math.transform import Transform, Rotation, Position

from phonebot.sim.common.model import *
from phonebot.sim.pybullet.urdf_editor import export_urdf


class PybulletBuilder(object):
    """
    Pybullet builder - translation layer between the generalized model definition
    to a format compatible with pybullet.
    """

    def __init__(self):
        self.joint_map_ = {ModelJointType.REVOLUTE: pb.JOINT_REVOLUTE,
                           ModelJointType.FIXED: pb.JOINT_FIXED,
                           ModelJointType.PRISMATIC: pb.JOINT_PRISMATIC}
        # Bookkeeping for input
        self.links_ = []
        self.joints_ = []
        # CreateMultibody() args
        self.model_ = {}
        # Bookkeeping for shape instances
        self.collision_shape_indices_ = {}
        self.visual_shape_indices_ = {}
        # Bookkeeping for cross name<->index referencing
        self.index_from_link_ = {}
        self.index_from_joint_ = {}
        self.link_from_index_ = {}
        self.joint_from_index_ = {}

    def _get_link_by_name(self, name: str) -> ModelLink:
        """ Temporary utility function to get a link by its name. """
        if name == self.root_.name:
            return self.root_
        for link in self.links_:
            if link.name == name:
                return link
        logging.warn('Link {} not found'.format(name))
        return None

    @lru_cache(maxsize=128)
    def _resolve_pose(self, pose: ModelPose) -> Transform:
        """ Resolve framed pose relative to the global root frame. """
        # Reached end!
        if pose.frame == self.root_.name:
            return pose.pose
        link = self._get_link_by_name(pose.frame)
        parent_pose = self._resolve_pose(link.pose)
        return parent_pose * pose.pose

    def _get_geom(self, geom: ModelGeometry,
                  pose: Transform, sim_id, col=False):
        """
        Instantiate pybullet geometry from a declarative `ModelGeometry` class.
        Deals with slightly different signatures/conventions while interfacing with pybullet.
        """
        cfun = partial(pb.createCollisionShape, physicsClientId=sim_id)
        vfun = partial(pb.createVisualShape, physicsClientId=sim_id)

        if isinstance(geom, ModelSphereGeometry):
            if col:
                return cfun(
                    pb.GEOM_SPHERE, radius=geom.radius,
                    collisionFramePosition=pose.position,
                    collisionFrameOrientation=pose.rotation.to_quaternion())
            else:
                return vfun(
                    pb.GEOM_SPHERE, radius=geom.radius,
                    visualFramePosition=pose.position,
                    visualFrameOrientation=pose.rotation.to_quaternion())
        if isinstance(geom, ModelBoxGeometry):
            if col:
                return cfun(
                    pb.GEOM_BOX, halfExtents=0.5 * geom.dimensions,
                    collisionFramePosition=pose.position,
                    collisionFrameOrientation=pose.rotation.to_quaternion())
            else:
                default_color = (0.0, 1.0, 1.0, 0.5)
                return vfun(
                    pb.GEOM_BOX, halfExtents=0.5 * geom.dimensions,
                    rgbaColor=default_color,
                    visualFramePosition=pose.position,
                    visualFrameOrientation=pose.rotation.to_quaternion()
                )
        if isinstance(geom, ModelCylinderGeometry):
            # quaternion to rotate z-axis->x-axis
            # basically means a canonical cylinder convention here
            # is pointing in the x-axis direction.
            qz_x = pb.getQuaternionFromAxisAngle([0, 1, 0], np.pi / 2)
            rel = Transform.from_rotation(Rotation.from_quaternion(qz_x))
            pose = pose * rel
            if col:
                return cfun(
                    pb.GEOM_CYLINDER, radius=geom.radius, height=geom.height,
                    collisionFramePosition=pose.position,
                    collisionFrameOrientation=pose.rotation.to_quaternion())
            else:
                return vfun(
                    pb.GEOM_CYLINDER, radius=geom.radius, length=geom.height,
                    visualFramePosition=pose.position,
                    visualFrameOrientation=pose.rotation.to_quaternion()
                )
        raise ValueError('Unsupported Geometry : {}'.format(geom))

    def _build_geom(self, sim_id):
        for link in [self.root_] + self.links_:
            geom = link.geometry
            if geom is None:
                continue

            # `geom` frame assumed to be attached to link pose.
            # Now that pybullet link position was moved to joint origin,
            # we need to re-set the geometry pose relative to joint origin.
            index = self.index_from_link_[link.name]
            if index >= 0:
                pose = (
                    self._resolve_pose(self.joints_[index].pose).inverse() *
                    self._resolve_pose(link.pose))
            else:
                pose = Transform.identity()

            if geom.name not in self.visual_shape_indices_:
                # NOTE(ycho): pose relative to link frame,
                # which is unfortunately the joint frame
                index = self._get_geom(geom, pose, sim_id)
                self.visual_shape_indices_[geom.name] = index
            if geom.name not in self.collision_shape_indices_:
                index = self._get_geom(geom, pose, sim_id, True)
                self.collision_shape_indices_[geom.name] = index

    def _resolve_root(self):
        links = [link.name for link in self.links_]
        childs = [joint.child for joint in self.joints_]
        roots = set.difference(set(links), set(childs))
        if len(roots) != 1:
            raise ValueError(
                'Number of possible root elements is not exactly one : {}'.format(
                    len(roots)))
        root = list(roots)[0]
        for link in self.links_:
            if link.name == root:
                self.root_ = link
                break

    def _add_root(self):
        link = self.root_
        self.model_['baseMass'] = link.mass
        self.model_['baseCollisionShapeIndex'] = link.geometry
        self.model_['baseVisualShapeIndex'] = link.geometry
        self.model_['basePosition'] = link.pose.pose.position
        self.model_['baseOrientation'] = link.pose.pose.rotation.to_quaternion()
        self.model_[
            'baseInertialFramePosition'] = link.inertia.pose.pose.position
        self.model_[
            'baseInertialFrameOrientation'] = link.inertia.pose.pose.rotation.to_quaternion()

    def _get_collision_shape_index(self, geometry):
        if geometry is None:
            return -1
        return self.collision_shape_indices_[geometry.name]

    def _get_visual_shape_index(self, geometry):
        if geometry is None:
            return -1
        return self.visual_shape_indices_[geometry.name]

    def _sort(self):
        links = [None for _ in range(len(self.links_))]
        joints = [None for _ in range(len(self.joints_))]
        for i in range(len(self.joints_)):
            joint = self.joints_[i]
            index = None
            for j in range(len(self.links_)):
                if self.links_[j].name == joint.child:
                    index = j
                    break

            assert not index is None, 'index for {} not found'.format(
                joints[i])
            links[i] = self.links_[index]
            joints[i] = self.joints_[i]

        self.links_ = links
        self.joints_ = joints

        # Joint <-> Index maps.
        self.index_from_link_ = {l.name: i for i, l in enumerate(self.links_)}
        self.index_from_link_[self.root_.name] = -1
        self.link_from_index_ = {v: k for k,
                                 v in self.index_from_link_.items()}
        self.index_from_joint_ = {
            j.name: i for i, j in enumerate(self.joints_)}
        self.joint_from_index_ = {v: k for k,
                                  v in self.index_from_joint_.items()}

    def add_link(self, link: ModelLink):
        self.links_.append(link)

    def _add_link(self, link: ModelLink):
        model = self.model_
        index = self.index_from_link_[link.name]
        joint = self.joints_[index]
        model['linkMasses'][index] = link.mass
        model['linkCollisionShapeIndices'][index] = link.geometry
        model['linkVisualShapeIndices'][index] = link.geometry

        # Get information about the parent to resolve relative transforms.
        parent_link = self._get_link_by_name(joint.parent)
        parent_index = self.index_from_link_[parent_link.name]
        parent_joint = self.joints_[parent_index]

        # NOTE(ycho): Parent pose convention deviates from pybullet documentation
        # that indicates joint frames are defined relative to
        # parent inertial frame - rather, I had to define the
        # joint pose relative to the parent link(=joint) frame.
        parent_pose = (Transform.identity() if parent_index <
                       0 else self._resolve_pose(parent_joint.pose))

        # Link pose relative to parent.
        joint_pose = self._resolve_pose(joint.pose)
        link_rel_pose = parent_pose.inverse() * joint_pose
        link_abs_pose = joint_pose  # Convenient alias.
        inertia_pose = link_abs_pose.inverse() * self._resolve_pose(link.inertia.pose)

        # NOTE(ycho): due to pybullet constraint, link pose == joint pose.
        model['linkPositions'][index] = link_rel_pose.position
        model['linkOrientations'][index] = link_rel_pose.rotation.to_quaternion()

        model['linkInertialFramePositions'][index] = inertia_pose.position
        model['linkInertialFrameOrientations'][index] = inertia_pose.rotation.to_quaternion()

    def add_joint(self, joint: ModelJoint):
        """ Add a joint to model specification. """
        self.joints_.append(joint)

    def _add_joint(self, joint: ModelJoint):
        model = self.model_
        index = self.index_from_joint_[joint.name]
        model['linkJointTypes'][index] = self.joint_map_[joint.type]
        axis = joint.axis
        model['linkJointAxis'][index] = axis
        model['linkParentIndices'][index] = self.index_from_link_[
            joint.parent] + 1

    def _preprocess(self):
        """ Fix missing information from the model specification """
        # Default inertial frame to link frame.
        for link in [self.root_] + self.links_:
            if link.inertia.pose.frame is None:
                logging.debug(
                    'Missing {} inertial frame set to link by default'.format(
                        link.name))
                link.inertia.pose.frame = link.name
            # Give unique geometry names.
            # NOTE(ycho): Necessary due to shape `instancing` no longer being possible
            # since each VisualShape/CollisionShape needs its own pose.
            link.geometry.name = '{}_{}'.format(link.name, link.geometry.name)

    def finalize(self):
        """
        Finalize model definition.
        """
        # Determine the root link.
        self._resolve_root()
        self.links_ = [link for link in self.links_ if (link != self.root_)]

        # Validation:
        # Number of links should be exactly identical to number of joints.
        # This is because our model definition is a tree - at least in
        # pybullet. Cycles are resolved in a model-specific postprocessing
        # step.
        if (len(self.links_) != len(self.joints_)):
            msg = '# Links != # joints : ({}!={})'.format(
                len(self.links_), len(self.joints_))
            raise ValueError(msg)

        # Process model specification to work with pybullet.
        self._preprocess()
        self._sort()

        # Finally, build the model (== args to createMultiBody())
        self.model_ = defaultdict(
            lambda: [None for _ in range(len(self.links_))])
        self._add_root()
        for link in self.links_:
            self._add_link(link)
        for joint in self.joints_:
            self._add_joint(joint)

        return dict(self.model_)

    def create(self, sim_id):
        """
        Instantiate the model configuration to online simulation.
        """
        model = self.model_

        # Add Geometry indices, which are only available
        # when connected to the backend simulator.
        self._build_geom(sim_id)

        # Remap to created indices from specifications.
        model['baseCollisionShapeIndex'] = self._get_collision_shape_index(
            self.root_.geometry)
        model['baseVisualShapeIndex'] = self._get_visual_shape_index(
            self.root_.geometry)
        model['linkCollisionShapeIndices'] = list(map(
            self._get_collision_shape_index,
            model['linkCollisionShapeIndices']))
        model['linkVisualShapeIndices'] = list(map(
            self._get_visual_shape_index,
            model['linkVisualShapeIndices']))

        # FIXME(yycho0108): hack to recover original link indices.
        # Pybullet does NOT preserve the specified order provided during
        # model creation, so an ugly workaround must be employed here
        # To figure out the original joint indices.
        masses = np.copy(model['linkMasses'])
        for i in range(len(masses)):
            model['linkMasses'][i] = i + 1

        # TODO(ycho): Enable after bullet3/PR#3238
        for i in range(len(masses)):
            model['linkNames'][i] = self.link_from_index_[i]

        # Actually create the model based on the temporary definition.
        robot = pb.createMultiBody(
            **self.model_,
            flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            physicsClientId=sim_id)

        # TODO(ycho): Enable after bullet3/PR#3238
        # for i in range(len(masses)):
        #     ji = pb.getJointInfo(robot, i, physicsClientId=sim_id)
        #     print('name[{}]={}'.format(i, ji[1]))

        # Recover the index map and restore the original mass properties.
        new_indices = np.arange(len(masses))
        old_indices = [int(np.round(pb.getDynamicsInfo(
            robot, i, physicsClientId=sim_id)[0] - 1)) for i in new_indices]
        for i0, i1 in zip(old_indices, new_indices):
            suc = pb.changeDynamics(
                robot, i1, mass=masses[i0], physicsClientId=sim_id)

        # Apply the new index permutations.
        joint_from_index = {}
        link_from_index = {}
        for i0, i1 in zip(old_indices, new_indices):
            link_from_index[i1] = self.link_from_index_[i0]
            joint_from_index[i1] = self.joint_from_index_[i0]

        # Cache the results and also store the inverse map.
        self.joint_from_index_ = joint_from_index
        self.link_from_index_ = link_from_index
        self.index_from_joint_ = {v: k for k,
                                  v in self.joint_from_index_.items()}
        self.index_from_link_ = {v: k for k,
                                 v in self.link_from_index_.items()}

        # Add the root body as well, which is considered separately
        # From other links. Again, this is mostly to follow bullet's
        # convention.
        # TODO(ycho): Hard-coding `body` here is probably not the best idea.
        self.index_from_link_['body'] = -1
        self.link_from_index_[-1] = 'body'
        self.robot_ = robot
        return robot
