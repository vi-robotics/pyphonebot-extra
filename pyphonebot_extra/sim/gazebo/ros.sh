#!/usr/bin/env bash
rosparam load '../../assets/gazebo/robot.urdf' /robot_description
rosparam load '../../assets/gazebo/control.yaml'
xterm -e 'roslaunch gazebo_ros empty_world.launch debug:=false physics:=ode paused:=true' &
sleep 5
rosrun gazebo_ros spawn_model -param '/robot_description' -unpause -model phonebot -urdf -z 0.5 \
    -J FL_hip_joint_a '-1.4' -J FL_hip_joint_b '-1.4' -J FR_hip_joint_a '-1.4' -J FR_hip_joint_b '-1.4' \
    -J HL_hip_joint_a '-1.4' -J HL_hip_joint_b '-1.4' -J HR_hip_joint_a '-1.4' -J HR_hip_joint_b '-1.4' \
    -J FL_knee_joint_a 2.4 -J FL_knee_joint_b 2.4 -J FR_knee_joint_a 2.4 -J FR_knee_joint_b 2.4 \
    -J HL_knee_joint_a 2.4 -J HL_knee_joint_b 2.4 -J HR_knee_joint_a 2.4 -J HR_knee_joint_b 2.4
xterm -e "ROS_NAMESPACE='/phonebot' rosrun controller_manager spawner joint_state_controller joints_position_controller" &
