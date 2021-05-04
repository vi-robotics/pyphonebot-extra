#!/usr/bin/env python3

import numpy as np
from phonebot.core.common.math.transform import Transform, Position, Rotation


def pose_from_text(pose: str) -> Transform:
    pose = np.fromstring(pose, sep=' ')
    pose = Transform(Position(pose[:3]), Rotation.from_euler(pose[3:]))
    return pose


def text_from_pose(pose: Transform) -> str:
    pose = list(pose.position) + list(pose.rotation.to_euler())
    return ' '.join(map(str, pose))
