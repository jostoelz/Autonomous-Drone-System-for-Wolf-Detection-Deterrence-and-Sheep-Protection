#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import random

def main():
    rospy.init_node("dynamic_object_node")
    pub = rospy.Publisher("/moving_object/pose", PoseStamped, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    # Startposition und Startgeschwindigkeit
    x, y, z = 0.0, 0.0, 1.0
    vx, vy = 0.1, 0.1

    while not rospy.is_shutdown():
        # Kleine zufällige Richtungsänderungen
        vx += random.uniform(-0.01, 0.01)
        vy += random.uniform(-0.01, 0.01)

        # Positionsupdate
        x += vx * 0.033  # 30 Hz -> dt ~ 0.033 s
        y += vy * 0.033

        # Begrenzung der Arena (-5m bis +5m)
        x = max(min(x, 5.0), -5.0)
        y = max(min(y, 5.0), -5.0)

        # Pose vorbereiten und veröffentlichen
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pub.publish(pose)

        rate.sleep()

if __name__ == "__main__":
    main()
