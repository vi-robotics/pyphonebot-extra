# Pybullet Simulator

Phonebot simulation implementation with pybullet backend.

For interfacing with `gym.Env` (and `gym.make()`), requires `register()` to be invoked, which is handled by the import (i.e. `from pyphonebot_extra.sim.pybullet.simulator import PybulletPhonebotEnv`).

## Running

```
python3 -m phonebot.app.demo_pybullet -h
pybullet build time: Jan 12 2020 16:28:02
usage: demo_pybullet.py [-h] [--sim.debug SIM.DEBUG]
                        [--sim.debug_axis_scale SIM.DEBUG_AXIS_SCALE]
                        [--sim.gravity SIM.GRAVITY]
                        [--sim.timestep SIM.TIMESTEP]
                        [--sim.render SIM.RENDER]
                        [--robot.leg_offset ROBOT.LEG_OFFSET]
                        [--robot.leg_sign ROBOT.LEG_SIGN]
                        [--robot.leg_rotation ROBOT.LEG_ROTATION]
                        [--robot.axis_sign ROBOT.AXIS_SIGN]
                        [--robot.hip_joint_offset ROBOT.HIP_JOINT_OFFSET]
                        [--robot.hip_link_length ROBOT.HIP_LINK_LENGTH]
                        [--robot.knee_link_length ROBOT.KNEE_LINK_LENGTH]
                        [--robot.body_dim ROBOT.BODY_DIM]
                        [--robot.leg_radius ROBOT.LEG_RADIUS]
                        [--robot.order ROBOT.ORDER]
                        [--robot.index ROBOT.INDEX]
                        [--robot.queue_size ROBOT.QUEUE_SIZE]
                        [--robot.timeout ROBOT.TIMEOUT]
                        [--robot.joint_names ROBOT.JOINT_NAMES]
                        [--robot.active_joint_names ROBOT.ACTIVE_JOINT_NAMES]
                        [--use_viewer USE_VIEWER]

optional arguments:
  -h, --help            show this help message and exit

:
  App settings for running pybullet simulator with phonebot.

  --use_viewer USE_VIEWER
                        use_viewer(<class 'bool'>) (default: True)

sim:

  --sim.debug SIM.DEBUG
                        debug(<class 'bool'>) (default: False)
  --sim.debug_axis_scale SIM.DEBUG_AXIS_SCALE
                        debug_axis_scale(<class 'float'>) (default: 0.02)
  --sim.gravity SIM.GRAVITY
                        gravity(<class 'float'>) (default: -9.81)
  --sim.timestep SIM.TIMESTEP
                        timestep(<class 'float'>) (default: 0.01618)
  --sim.render SIM.RENDER
                        render(<class 'bool'>) (default: True)

robot:
  Parametric phonebot configuration.

  --robot.leg_offset ROBOT.LEG_OFFSET
                        leg_offset(typing.Tuple[float, float, float])
                        (default: (0.041418, 0.0425, -0.010148))
  --robot.leg_sign ROBOT.LEG_SIGN
                        leg_sign(typing.List[typing.Tuple[int, int, int]])
                        (default: [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1,
                        1)])
  --robot.leg_rotation ROBOT.LEG_ROTATION
                        leg_rotation(typing.List[phonebot.core.common.math.tra
                        nsform.Rotation]) (default: (Rotation([ 1. , 0. , 0. ,
                        -1.57079633])(rtype=axis_angle),
                        Rotation([-4.32978028e-17, -7.07106781e-01,
                        7.07106781e-01, 4.32978028e-17])(rtype=quaternion),
                        Rotation([ 1. , 0. , 0. ,
                        -1.57079633])(rtype=axis_angle),
                        Rotation([-4.32978028e-17, -7.07106781e-01,
                        7.07106781e-01, 4.32978028e-17])(rtype=quaternion)))
  --robot.axis_sign ROBOT.AXIS_SIGN
                        axis_sign(typing.Tuple[int, int, int]) (default: (1,
                        1, 1, 1))
  --robot.hip_joint_offset ROBOT.HIP_JOINT_OFFSET
                        hip_joint_offset(<class 'float'>) (default: 0.011)
  --robot.hip_link_length ROBOT.HIP_LINK_LENGTH
                        hip_link_length(<class 'float'>) (default: 0.0175)
  --robot.knee_link_length ROBOT.KNEE_LINK_LENGTH
                        knee_link_length(<class 'float'>) (default: 0.0285)
  --robot.body_dim ROBOT.BODY_DIM
                        body_dim(typing.Tuple[float, float, float]) (default:
                        [0.15 0.07635 0.021612])
  --robot.leg_radius ROBOT.LEG_RADIUS
                        leg_radius(<class 'float'>) (default: 0.004)
  --robot.order ROBOT.ORDER
                        order(typing.Tuple[str, str, str, str]) (default:
                        ('FL', 'FR', 'HL', 'HR'))
  --robot.index ROBOT.INDEX
                        index(typing.Dict[str, int]) (default: {'FL': 0, 'FR':
                        1, 'HL': 2, 'HR': 3})
  --robot.queue_size ROBOT.QUEUE_SIZE
                        queue_size(<class 'int'>) (default: 4)
  --robot.timeout ROBOT.TIMEOUT
                        timeout(<class 'float'>) (default: 0.1)
  --robot.joint_names ROBOT.JOINT_NAMES
                        joint_names(typing.List[str]) (default:
                        ['FL_knee_joint_a', 'FL_hip_joint_a',
                        'FL_knee_joint_b', 'FL_hip_joint_b',
                        'FR_knee_joint_a', 'FR_hip_joint_a',
                        'FR_knee_joint_b', 'FR_hip_joint_b',
                        'HL_knee_joint_a', 'HL_hip_joint_a',
                        'HL_knee_joint_b', 'HL_hip_joint_b',
                        'HR_knee_joint_a', 'HR_hip_joint_a',
                        'HR_knee_joint_b', 'HR_hip_joint_b'])
  --robot.active_joint_names ROBOT.ACTIVE_JOINT_NAMES
                        active_joint_names(typing.List[str]) (default:
                        ['FL_hip_joint_a', 'FL_hip_joint_b', 'FR_hip_joint_a',
                        'FR_hip_joint_b', 'HL_hip_joint_a', 'HL_hip_joint_b',
                        'HR_hip_joint_a', 'HR_hip_joint_b'])
```
