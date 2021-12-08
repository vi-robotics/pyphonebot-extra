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


class PybulletBuilder(object):
    """
    Pybullet builder - translation layer between the generalized model definition
    to a format compatible with pybullet.
    """

    def __init__(self):
        self._joint_map = {ModelJointType.REVOLUTE: pb.JOINT_REVOLUTE,
                           ModelJointType.FIXED: pb.JOINT_FIXED,
                           ModelJointType.PRISMATIC: pb.JOINT_PRISMATIC}
        # Bookkeeping for input
        self._links = []
        self._joints = []
        # CreateMultibody() args
        self._model = {}
        # Bookkeeping for shape instances
        self._collision_shape_indices = {}
        self._visual_shape_indices = {}
        # Bookkeeping for cross name<->index referencing
        self._index_from_link = {}
        self._index_from_joint = {}
        self._link_from_index = {}
        self._joint_from_index = {}

    def _get_link_by_name(self, name: str) -> ModelLink:
        """Utility funciton to get a link from a name

        Args:
            name (str): The name of the link

        Returns:
            ModelLink: The model link
        """
        if name == self._root.name:
            return self._root
        for link in self._links:
            if link.name == name:
                return link
        logging.warn('Link {} not found'.format(name))
        return None

    @lru_cache(maxsize=128)
    def _resolve_pose(self, pose: ModelPose) -> Transform:
        """Resolve framed pose relative to the global root frame.

        Args:
            pose (ModelPose): The relative model pose

        Returns:
            Transform: The global transform of the pose
        """
        # Reached end!
        if pose.frame == self._root.name:
            return pose.pose
        link = self._get_link_by_name(pose.frame)
        parent_pose = self._resolve_pose(link.pose)
        return parent_pose * pose.pose

    def _get_geom(self, geom: ModelGeometry,
                  pose: Transform, sim_id: int, col: bool = False):
        """Instantiate pybullet geometry from a declarative `ModelGeometry`
        class. Deals with slightly different signatures/conventions while
        interfacing with pybullet.

        Args:
            geom (ModelGeometry): The geometry to instantiate
            pose (Transform): The relative pose of the geometry to the link
                frame.
            sim_id (int): The id of the physics client server
            col (bool, optional): If True, then return a collision shape. Else,
                return a visual geometry. Defaults to False.

        Raises:
            ValueError: Unsupported Geometry

        Returns:
            functools.partial: A callable to get the collision or visual shape
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

    def _build_geom(self, sim_id: int):
        """Build the model in a given physics client

        Args:
            sim_id (int): The id of the physics client server.
        """
        for link in [self._root] + self._links:
            geom = link.geometry
            if geom is None:
                continue

            # `geom` frame assumed to be attached to link pose.
            # Now that pybullet link position was moved to joint origin,
            # we need to re-set the geometry pose relative to joint origin.
            index = self._index_from_link[link.name]
            if index >= 0:
                pose = (
                    self._resolve_pose(self._joints[index].pose).inverse() *
                    self._resolve_pose(link.pose))
            else:
                pose = Transform.identity()

            if geom.name not in self._visual_shape_indices:
                # NOTE(ycho): pose relative to link frame,
                # which is unfortunately the joint frame
                index = self._get_geom(geom, pose, sim_id)
                self._visual_shape_indices[geom.name] = index
            if geom.name not in self._collision_shape_indices:
                index = self._get_geom(geom, pose, sim_id, True)
                self._collision_shape_indices[geom.name] = index

    def _resolve_root(self):
        """The root is a link with no parent. This sets the root to the current
        root link.

        Raises:
            ValueError: Number of possible root elements is not exactly one.
        """
        links = [link.name for link in self._links]
        childs = [joint.child for joint in self._joints]
        roots = set.difference(set(links), set(childs))
        if len(roots) != 1:
            raise ValueError(
                'Number of possible root elements is not exactly one : {}'.format(
                    len(roots)))
        root = list(roots)[0]
        for link in self._links:
            if link.name == root:
                self._root = link
                break

    def _add_root(self):
        """Sets the current root of the model
        """
        link = self._root
        self._model['baseMass'] = link.mass
        self._model['baseCollisionShapeIndex'] = link.geometry
        self._model['baseVisualShapeIndex'] = link.geometry
        self._model['basePosition'] = link.pose.pose.position
        self._model['baseOrientation'] = link.pose.pose.rotation.to_quaternion()
        self._model[
            'baseInertialFramePosition'] = link.inertia.pose.pose.position
        self._model[
            'baseInertialFrameOrientation'] = link.inertia.pose.pose.rotation.to_quaternion()

    def _get_collision_shape_index(self, geometry: ModelGeometry) -> int:
        """Returns the index of the collision model geometry

        Args:
            geometry (ModelGeometry): The model geometry of the collision shape

        Returns:
            int: The index of the collision shape
        """
        if geometry is None:
            return -1
        return self._collision_shape_indices[geometry.name]

    def _get_visual_shape_index(self, geometry: ModelGeometry) -> int:
        """Returns the index of the visual model geometry

        Args:
            geometry (ModelGeometry): The model geometry of the visual shape

        Returns:
            int: The index of the visual shape
        """
        if geometry is None:
            return -1
        return self._visual_shape_indices[geometry.name]

    def _sort(self):
        """Sort the links and joints by parent-child relationships
        """
        links = [None for _ in range(len(self._links))]
        joints = [None for _ in range(len(self._joints))]
        for i in range(len(self._joints)):
            joint = self._joints[i]
            index = None
            for j in range(len(self._links)):
                if self._links[j].name == joint.child:
                    index = j
                    break

            assert not index is None, 'index for {} not found'.format(
                joints[i])
            links[i] = self._links[index]
            joints[i] = self._joints[i]

        self._links = links
        self._joints = joints

        # Joint <-> Index maps.
        self._index_from_link = {l.name: i for i, l in enumerate(self._links)}
        self._index_from_link[self._root.name] = -1
        self._link_from_index = {v: k for k,
                                 v in self._index_from_link.items()}
        self._index_from_joint = {
            j.name: i for i, j in enumerate(self._joints)}
        self._joint_from_index = {v: k for k,
                                  v in self._index_from_joint.items()}

    def add_link(self, link: ModelLink):
        """Add a link to the builder list of links

        Args:
            link (ModelLink): The model link to add
        """
        self._links.append(link)

    def _add_link(self, link: ModelLink):

        model = self._model
        index = self._index_from_link[link.name]
        joint = self._joints[index]
        model['linkMasses'][index] = link.mass
        model['linkCollisionShapeIndices'][index] = link.geometry
        model['linkVisualShapeIndices'][index] = link.geometry

        # Get information about the parent to resolve relative transforms.
        parent_link = self._get_link_by_name(joint.parent)
        parent_index = self._index_from_link[parent_link.name]
        parent_joint = self._joints[parent_index]

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
        """Add a joint to model specification.


        Args:
            joint (ModelJoint): The joint to add to the model.
        """
        self._joints.append(joint)

    def _add_joint(self, joint: ModelJoint):
        model = self._model
        index = self._index_from_joint[joint.name]
        model['linkJointTypes'][index] = self._joint_map[joint.type]
        axis = joint.axis
        model['linkJointAxis'][index] = axis
        model['linkParentIndices'][index] = self._index_from_link[
            joint.parent] + 1

    def _preprocess(self):
        """Fix missing information from the model specification
        """
        # Default inertial frame to link frame.
        for link in [self._root] + self._links:
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

        Validation:
            Number of links should be exactly identical to number of joints.
            This is because our model definition is a tree - at least in
            pybullet. Cycles are resolved in a model-specific postprocessing
            step.
        """
        # Determine the root link.
        self._resolve_root()
        self._links = [link for link in self._links if (link != self._root)]

        if (len(self._links) != len(self._joints)):
            msg = '# Links != # joints : ({}!={})'.format(
                len(self._links), len(self._joints))
            raise ValueError(msg)

        # Process model specification to work with pybullet.
        self._preprocess()
        self._sort()

        # Finally, build the model (== args to createMultiBody())
        self._model = defaultdict(
            lambda: [None for _ in range(len(self._links))])
        self._add_root()
        for link in self._links:
            self._add_link(link)
        for joint in self._joints:
            self._add_joint(joint)

        return dict(self._model)

    def create(self, sim_id: int) -> int:
        """Instantiate the model configuration to online simulation.

        Args:
            sim_id (int): The id of the physics client server.

        Returns:
            int: The unique id of the robot body
        """
        model = self._model

        # Add Geometry indices, which are only available
        # when connected to the backend simulator.
        self._build_geom(sim_id)

        # Remap to created indices from specifications.
        model['baseCollisionShapeIndex'] = self._get_collision_shape_index(
            self._root.geometry)
        model['baseVisualShapeIndex'] = self._get_visual_shape_index(
            self._root.geometry)
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
            model['linkNames'][i] = self._link_from_index[i]

        # Actually create the model based on the temporary definition.
        robot = pb.createMultiBody(
            **self._model,
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
            link_from_index[i1] = self._link_from_index[i0]
            joint_from_index[i1] = self._joint_from_index[i0]

        # Cache the results and also store the inverse map.
        self._joint_from_index = joint_from_index
        self._link_from_index = link_from_index
        self._index_from_joint = {v: k for k,
                                  v in self._joint_from_index.items()}
        self._index_from_link = {v: k for k,
                                 v in self._link_from_index.items()}

        # Add the root body as well, which is considered separately
        # From other links. Again, this is mostly to follow bullet's
        # convention.
        # TODO(ycho): Hard-coding `body` here is probably not the best idea.
        self._index_from_link['body'] = -1
        self._link_from_index[-1] = 'body'
        self.robot_ = robot
        return robot
