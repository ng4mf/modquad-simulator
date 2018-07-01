import math

import tf
from geometry_msgs.msg import PoseStamped
import rospy
from std_srvs.srv import Empty


def goal_to_pose(x, y, z, yaw):
    goal = PoseStamped()
    goal.header.seq = 0
    goal.header.frame_id = '/world'

    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
    print quaternion
    goal.pose.orientation.x = quaternion[0]
    goal.pose.orientation.y = quaternion[1]
    goal.pose.orientation.z = quaternion[2]
    goal.pose.orientation.w = quaternion[3]
    return goal


def circular_motion():
    rospy.init_node('circular', anonymous=True)
    n = 5
    # Prefix
    prefix = 'modquad'

    # Goal publishers
    publishers = [rospy.Publisher('/%s0%d/goal' % (prefix, i + 1), PoseStamped, queue_size=1) for i in range(n)]
    # Takeoff service
    takeoff_services = [rospy.ServiceProxy('/%s0%d/takeoff' % (prefix, i + 1), Empty) for i in range(n)]

    # takeoff for all robots
    for takeoff in takeoff_services:
        takeoff()

    # Time counter
    t = 1.
    s = 100.
    # Circle loop
    while not rospy.is_shutdown():
        for i in range(n):
            theta = t / s + i * 2 * math.pi / n
            publishers[i].publish(goal_to_pose(math.cos(theta), math.sin(theta), 1, 0.))

        t += 1
        rospy.sleep(.1)


if __name__ == '__main__':
    circular_motion()
