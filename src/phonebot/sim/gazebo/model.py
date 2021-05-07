# uncompyle6 version 3.5.1
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Oct  8 2019, 13:06:37)
# [GCC 5.4.0 20160609]
# Embedded file name: /home/jamiecho/Repos/PhoneBot/control/sim/gazebo/model.py
# Compiled at: 2019-11-19 21:24:08
# Size of source mod 2**32: 20171 bytes

from abc import abstractmethod
from xml.etree import ElementTree as et
from xml.dom import minidom
import sys
from collections import namedtuple, defaultdict, deque

import time
import pybullet as p
import numpy as np

from phonebot.core.common.math.transform import Transform, Rotation, Position
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.utils import anorm
from phonebot.core.controls.controllers.trajectory_controller import EndpointTrajectoryGraphController
from phonebot.core.frame_graph.graph_utils import solve_inverse_kinematics, solve_inverse_kinematics_half
from phonebot.core.controls.agents.trajectory_agent import TrajectoryAgentGraph
from phonebot.core.frame_graph.frame_edges import StaticFrameEdge
from phonebot.core.frame_graph.frame_graph import FrameGraph
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph import get_graph_geometries, FrameGraph
from phonebot.vis.viewer import PhonebotViewer
from phonebot.vis.viewer.proxy_command import ProxyCommand
from phonebot.vis.viewer.proxy_commands import AddLineStripCommand
from sim.common.model import *


class XmlInertia(et.Element):

    @classmethod
    def from_matrix(cls, inertia):
        self = cls('inertia')
        xyz = 'xyz'
        for i in range(3):
            for j in range(i, 3):
                name = 'i{}{}'.format(xyz[i], xyz[j])
                et.SubElement(self, name).text = str(inertia[(i, j)])

        return self

    @classmethod
    def from_sphere(cls, mass, radius):
        imat = 0.4 * m * r * r * np.eye(3)
        return cls.from_matrix(imat)

    @classmethod
    def from_box(cls, mass, dimensions):
        dim_sq = np.square(dimensions)
        imat = 0.08333333333333333 * mass * np.diag(dim_sq.sum() - dim_sq)
        return cls.from_matrix(imat)


class XmlPose(et.Element):

    @classmethod
    def from_transform(cls, pose: Transform):
        x, y, z = pose.position
        rx, ry, rz = pose.rotation.to_euler()
        return cls.from_vector6([x, y, z, rx, ry, rz])

    @classmethod
    def from_vector6(cls, pose):
        self = cls('pose')
        self.text = ' '.join(map(str, pose))
        return self


def create_capsule(radius, length, col):
    rot = p.getQuaternionFromAxisAngle([0, 1, 0], np.pi / 2)
    vals = [p.GEOM_CAPSULE, radius, length, rot, [length / 2, 0, 0]]
    if col:
        fun = p.createCollisionShape
        args = ['shapeType', 'radius', 'height',
                'collisionFrameOrientation', 'collisionFramePosition']
    else:
        fun = p.createVisualShape
        args = ['shapeType', 'radius', 'length',
                'visualFrameOrientation', 'visualFramePosition']
    return fun(**{a: v for a, v in zip(args, vals)})


class XmlLink(object):

    def __init__(self):
        pass


class SdfBuilder(object):

    def __init__(self):
        self.root_ = et.Element('sdf', version='1.6')
        self.model_ = et.SubElement(self.root_, 'model', name='phonebot')
        et.SubElement(self.model_, 'self_collide').text = 'false'

    def postprocess(self):
        joints_to_kill = set()
        links_to_kill = {}
        for joint in self.model_.iter('joint'):
            if joint.get('type').lower() != 'fixed':
                continue
            joints_to_kill.add(joint)
            parent_frame = joint.find('parent').text
            child_frame = joint.find('child').text
            links_to_kill[child_frame] = parent_frame

        for link in links_to_kill:
            frame = links_to_kill[link]
            pose = Transform.identity()
            while 1:
                if frame in links_to_kill:
                    lpose = pose_from_text(
                        self.get_link(frame).find('pose').text)
                    pose = lpose * pose
                    frame = links_to_kill[frame]

            parent = self.get_link(frame)
            link = self.get_link(link)
            for gtype in ['visual', 'collision']:
                cel = link.find(gtype)
                new_name = '{}_{}'.format(link.get('name'), cel.get('name'))
                cel.set('name', new_name)
                new_pose = pose * pose_from_text(cel.find('pose').text)
                cel.find('pose').text = text_from_pose(new_pose)
                link.remove(cel)
                parent.append(cel)

        for joint in joints_to_kill:
            self.model_.remove(joint)

    def add_pose(self, elem, pose, frame=None):
        """
        Add 6dof pose to element.
        """
        if frame is not None:
            frame_pose = self.get_link(frame).find('pose').text
            p0 = pose_from_text(frame_pose)
            p1 = Transform(pose[:3], Rotation.from_euler(pose[3:]))
            p0p1 = p0 * p1
            pose = list(p0p1.position) + list(p0p1.rotation.to_euler())
        elem = et.SubElement(elem, 'pose')
        elem.text = ' '.join(map(str, pose))

    @staticmethod
    def add_axis(elem, axis):
        elem = et.SubElement(elem, 'axis')
        et.SubElement(elem, 'xyz').text = ' '.join(map(str, axis))

    def add_inertia(self, link, pose, mass, inertia):
        """
        Add 3x3 inertia matrix to link element.
        """
        elem = et.SubElement(link, 'inertial')
        self.add_pose(elem, pose)
        et.SubElement(elem, 'mass').text = str(mass)
        elem = et.SubElement(elem, 'inertia')
        xyz = 'xyz'
        for i in range(3):
            for j in range(i, 3):
                name = 'i{}{}'.format(xyz[i], xyz[j])
                et.SubElement(elem, name).text = str(inertia[(i, j)])

    def add_link(self, name, pose, mass, inertia, geom=None, parent=None):
        """
        :D
        """
        link = et.SubElement(self.model_, 'link', name=name)
        frame = None
        if parent is not None:
            frame = parent.get('name')
        self.add_pose(link, pose, frame=frame)
        self.add_inertia(link, [0, 0, 0, 0, 0, 0], mass, inertia)
        if geom is not None:
            geom_pose, geom_elem = geom
            for gtype in ['visual', 'collision']:
                geom_pose, geom_elem = geom
                elem = et.SubElement(link, gtype, name=gtype)
                self.add_pose(elem, geom_pose)
                elem.append(geom_elem)

    def add_joint(self, name, type, parent, child, pose, axis):
        joint = et.SubElement(self.model_, 'joint', name=name, type=type)
        et.SubElement(joint, 'parent').text = parent
        et.SubElement(joint, 'child').text = child
        self.add_pose(joint, pose)
        self.add_axis(joint, axis)

    def get_link(self, name):
        for link in self.model_.iter('link'):
            if link.get('name') == name:
                return link


def main():
    use_viewer = True
    config = PhonebotSettings()
    graph = PhonebotGraph(config)
    if use_viewer:
        data_queue, event_queue, command_queue = PhonebotViewer.create()
        for leg_prefix in config.order:
            command_queue.put(AddLineStripCommand(
                name='{}_trajectory'.format(leg_prefix)))

        foot_positions = defaultdict(lambda: deque(maxlen=256))
    delta_t = 0.01618
    nsolve = 1024
    sim_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)
    plane = p.createCollisionShape(p.GEOM_PLANE)
    pidx = p.createMultiBody(0, plane)
    p.resetBasePositionAndOrientation(pidx, [0, 0, -0.15], [0, 0, 0, 1])
    joints, links = PhonebotBuilder().build()
    builder = PybulletBuilder()
    for joint in joints:
        builder.add_joint(joint)

    for link in links:
        builder.add_link(link)

    model = builder.finalize()
    phonebot_id = builder.create()
    axs = 0.02
    for prefix in config.order:
        for suffix in 'ab':
            for jtype in ('knee', 'hip'):
                jname = '{}_{}_link_{}'.format(prefix, jtype, suffix)
                for ax in np.eye(3):
                    p.addUserDebugLine([
                        0, 0, 0], axs * ax, ax, parentObjectUniqueId=phonebot_id, parentLinkIndex=builder.index_from_link_[jname])

    p.addUserDebugText('phonebot', [0, 0, 0.2], [
                       0, 0, 0], parentObjectUniqueId=phonebot_id, parentLinkIndex=builder.index_from_link_['body'])
    for prefix in config.order:
        for suffix in 'ab':
            index = builder.index_from_joint_[
                '{}_hip_joint_{}'.format(prefix, suffix)]
            p.resetJointState(phonebot_id, index, -1.4)
            index = builder.index_from_joint_[
                '{}_knee_joint_{}'.format(prefix, suffix)]
            print('index', index)
            p.resetJointState(phonebot_id, index, 2.4)

    for prefix in config.order:
        ia = builder.index_from_link_['{}_foot_link_a'.format(prefix)]
        ib = builder.index_from_link_['{}_foot_link_b'.format(prefix)]
        c = p.createConstraint(phonebot_id, ia, phonebot_id, ib, p.JOINT_POINT2POINT, [
            0.0, 0.0, 1.0], [
            0, 0, 0], [0, 0, 0])
        # NOTE(yycho0108): is this meainingful??
        # p.changeConstraint(c, maxForce=3.76)

    joint_edges = []
    for prefix in config.order:
        for suffix in 'ab':
            knee = '{}_knee_joint_{}'.format(prefix, suffix)
            hip = '{}_hip_joint_{}'.format(prefix, suffix)
            joint_edge = graph.get_edge(knee, hip)
            joint_edges.append(joint_edge)

    agent = TrajectoryAgentGraph(graph, 4.0, config)
    urdf = UrdfEditor()
    urdf.initializeFromBulletBody(phonebot_id, sim_id)
    text = urdf.to_string()
    sremap = {'link0': 'body'}
    for i in reversed(range(p.getNumJoints(phonebot_id))):
        info = p.getJointInfo(phonebot_id, i)
        old_jname, old_lname = info[1].decode(
            'utf-8'), info[12].decode('utf-8')
        new_jname = builder.joint_from_index_[i]
        new_lname = builder.link_from_index_[i]
        sremap[old_jname] = new_jname
        sremap[old_lname] = new_lname

    for old_name in reversed(sorted(sremap.keys())):
        new_name = sremap[old_name]
        text = text.replace(old_name, new_name)

    with open('/tmp/phonebot.urdf', 'w') as (f):
        f.write(text)
    h = 0
    stamp = 0.0
    while True:
        stamp += delta_t
        print('stamp : {}'.format(stamp))
        js = {}
        keys = []
        for prefix in config.order:
            for suffix in 'ab':
                for jtype in ('knee', 'hip'):
                    jname = '{}_{}_joint_{}'.format(prefix, jtype, suffix)
                    keys.append(jname)

        indices = [builder.index_from_joint_[k] for k in keys]
        states = p.getJointStates(phonebot_id, indices)
        js = {k: s[0] for k, s in zip(keys, states)}
        for prefix in config.order:
            for suffix in 'ab':
                hip_joint = '{}_hip_joint_{}'.format(prefix, suffix)
                knee_joint = '{}_knee_joint_{}'.format(prefix, suffix)
                foot_joint = '{}_foot_{}'.format(prefix, suffix)
                graph.get_edge(knee_joint, hip_joint).update(
                    stamp, js[hip_joint])
                graph.get_edge(foot_joint, knee_joint).update(
                    stamp, js[knee_joint])

        commands = agent(stamp)
        for edge, command in zip(joint_edges, commands):
            p.setJointMotorControl2(phonebot_id, builder.index_from_joint_[
                                    edge.target], p.POSITION_CONTROL, targetPosition=command, force=0.235)

        if use_viewer:
            poses, edges = get_graph_geometries(graph, stamp, tol=np.inf)
            visdata = {'poses': dict(poses=poses), 'edges': dict(
                poses=poses, edges=edges)}
            for leg_prefix in config.order:
                foot_joint = '{}_foot_a'.format(leg_prefix)
                foot_positions[leg_prefix].append(
                    graph.get_transform(foot_joint, 'local', stamp).position)

            for leg_prefix in config.order:
                tag = '{}_trajectory'.format(leg_prefix)
                visdata[tag] = dict(pos=np.asarray(foot_positions[leg_prefix]), color=(0.0,
                                                                                  1.0,
                                                                                  1.0,
                                                                                  1.0))

            if not data_queue.full():
                data_queue.put_nowait(visdata)
        p.stepSimulation()


if __name__ == '__main__':
    main()
# okay decompiling model.cpython-35.pyc
