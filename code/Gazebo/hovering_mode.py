#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Point, PoseStamped

class HoveringVelocity:
    def __init__(self):
        # Node
        rospy.init_node("hovering_node")

        # Publisher für Velocity
        self.vel_pub = rospy.Publisher("/hovering_velocity", Twist, queue_size=10)

        # Subscriber für State
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        self.current_position = Point()
        self.current_position.z = 0.0
        self.altitude = 2.0
        self.max_vel = 3.0

        #  Controller Gains
        self.kp_z = 0.5

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_position.z = msg.pose.position.z

    def compute_velocity(self):
        dz = self.altitude - self.current_position.z

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = max(min(dz * self.kp_z, self.max_vel), -self.max_vel)
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.y = 0
        return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()

        # Warte auf Verbindung zum FCU
        while not rospy.is_shutdown() and (self.current_state is None or not self.current_state.connected):
            rate.sleep()

        while not rospy.is_shutdown():
            # Velocity berechnen und senden
            twist = self.compute_velocity()
            self.vel_pub.publish(twist)
            rate.sleep()

if __name__ == "__main__":
    controller = HoveringVelocity()
    controller.run()
