#!/usr/bin/env python3

from phonebot.core.common.path import PhonebotPath

import yaml
import rospy
import roslaunch


class GazeboEnv(object):
    """
    Gazebo env, deprecated - develpment is  inactive for now.
    """

    def __init__(self):
        pass

    def start(self):
        assets = PhonebotPath.assets()
        urdf = os.path.join(assets, 'gazebo', 'robot.urdf')
        ctrl = os.path.join(assets, 'gazebo', 'control.yaml')
        with open(urdf, 'r') as f:
            rospy.set_param('/phonebot/robot_description', yaml.load(f))
        with open(ctrl, 'r') as f:
            rospy.set_param('/phonebot/control', yaml.load(f))
        # roslaunch gazebo_ros empty_world.launch debug:=true physics:=ode
        # rosparam load control.yaml
        # ROS_NAMESPACE='/robot' rosrun controller_manager spawner joint_state_controller joints_position_controller
        # rostopic pub ...
        pass

    def run(self):
        self.start()
        pass


def main():
    env = GazeboEnv()
    env.run()
    pass


if __name__ == '__main__':
    main()
