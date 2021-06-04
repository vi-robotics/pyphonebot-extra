#!/usr/bin/env python3

from typing import List
import pybullet as pb
import numpy as np
import math


def debug_get_full_aabb(sim_id: int, robot_id: int) -> np.ndarray:
    """Get full AABB of a robot including all of its constituent links.

    Args:
        sim_id (int): The id of the pybullet sim
        robot_id (int): The id of the pybullet robot

    Returns:
        np.ndarray: An #2x3 array representing the min and max corners of the
            axis-aligned bounding box
    """
    num_joints = pb.getNumJoints(robot_id, physicsClientId=sim_id)
    aabb = np.asarray(pb.getAABB(
        robot_id, -1, physicsClientId=sim_id), dtype=np.float32)
    for i in range(num_joints):
        link_aabb = pb.getAABB(robot_id, i, physicsClientId=sim_id)
        aabb[0] = np.minimum(aabb[0], link_aabb[0], out=aabb[0])
        aabb[1] = np.maximum(aabb[1], link_aabb[1], out=aabb[1])
    return aabb


def debug_draw_frame_axes(sim_id: int,
                          robot_id: int,
                          scale: float = 1.0,
                          joint_indices: List[int] = None) -> None:
    """Draw XYZ frame axes over links (color: RGB) If `joint_indices` is not
    specified, frames will be drawn for all joints.

    Args:
        sim_id (int): The id of the pybullet sim
        robot_id (int): The id of the pybullet robot
        scale (float, optional): By default, debug lines have unit length. Scale
            is multilplied by this length to display the axes. Defaults to 1.0.
        joint_indices (List[int], optional): The indices of the joints to
            draw debug axes for in the pybullet sim. Defaults to None.
    """
    # If joint indices not specified, draw over all existing links.
    if joint_indices is None:
        joint_indices = range(pb.getNumJoints(
            robot_id, physicsClientId=sim_id))

    for ji in joint_indices:
        for ax in np.eye(3):
            pb.addUserDebugLine([0, 0, 0], scale * ax, ax,
                                parentObjectUniqueId=robot_id,
                                parentLinkIndex=ji, physicsClientId=sim_id)


def debug_draw_inertia_box(parent_uid: int, parent_link_index: int, color: List[int]):
    """Draw the inertia box around a given link

    Args:
        parent_uid (int): The id of the parent of the link to draw the debug
            entity in local coordinates
        parent_link_index (int): The index of the link in the parent
        color (List[int]): A length 3 list of red, green, blue values

    (taken from pybullet/examples/quadruped.py)
    """
    dyn = pb.getDynamicsInfo(parent_uid, parent_link_index)
    mass = dyn[0]
    inertia = dyn[2]
    if (mass > 0):
        Ixx = inertia[0]
        Iyy = inertia[1]
        Izz = inertia[2]
        boxScaleX = 0.5 * math.sqrt(6 * (Izz + Iyy - Ixx) / mass)
        boxScaleY = 0.5 * math.sqrt(6 * (Izz + Ixx - Iyy) / mass)
        boxScaleZ = 0.5 * math.sqrt(6 * (Ixx + Iyy - Izz) / mass)

        halfExtents = [boxScaleX, boxScaleY, boxScaleZ]
        pts = [[halfExtents[0], halfExtents[1], halfExtents[2]],
               [-halfExtents[0], halfExtents[1], halfExtents[2]],
               [halfExtents[0], -halfExtents[1], halfExtents[2]],
               [-halfExtents[0], -halfExtents[1], halfExtents[2]],
               [halfExtents[0], halfExtents[1], -halfExtents[2]],
               [-halfExtents[0], halfExtents[1], -halfExtents[2]],
               [halfExtents[0], -halfExtents[1], -halfExtents[2]],
               [-halfExtents[0], -halfExtents[1], -halfExtents[2]]]

        pb.addUserDebugLine(pts[0],
                            pts[1],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[1],
                            pts[3],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[3],
                            pts[2],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[2],
                            pts[0],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)

        pb.addUserDebugLine(pts[0],
                            pts[4],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[1],
                            pts[5],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[2],
                            pts[6],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[3],
                            pts[7],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)

        pb.addUserDebugLine(pts[4 + 0],
                            pts[4 + 1],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[4 + 1],
                            pts[4 + 3],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[4 + 3],
                            pts[4 + 2],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
        pb.addUserDebugLine(pts[4 + 2],
                            pts[4 + 0],
                            color,
                            1,
                            parentObjectUniqueId=parent_uid,
                            parentLinkIndex=parent_link_index)
