import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


def main():
    rospy.init_node('phonebot_stand', anonymous=True)
    cmd_topic = '/phonebot/joints_position_controller/command'
    pub = rospy.Publisher(cmd_topic, Float64MultiArray, queue_size=16)
    msg = Float64MultiArray()
    msg.layout.dim.append(
        MultiArrayDimension(
            label='',
            size=8,
            stride=8))

    rate = rospy.Rate(10)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        if t0.is_zero():
            t0 = rospy.Time.now()
            continue
        t1 = rospy.Time.now()
        dt = (t1-t0).to_sec()
        cmd = np.clip(dt / 10.0, 0.0, 1.0) * np.full(8, -1.3)
        msg.data = cmd
        pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    main()
