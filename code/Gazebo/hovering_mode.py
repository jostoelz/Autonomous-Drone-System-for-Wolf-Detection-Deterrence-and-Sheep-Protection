#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Point, PoseStamped

class HoveringVelocity:
    def __init__(self):
        # Node initialization
        rospy.init_node("hovering_node")

        # Publisher for sending velocity commands
        self.vel_pub = rospy.Publisher("/hovering_velocity", Twist, queue_size=10)

        # Subscriber for vehicle state information
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()
        
        # Subscriber for local position to monitor altitude
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        # Initialization of position and parameter variables
        self.current_position = Point()
        self.current_position.z = 0.0
        self.altitude = 2.0  # Target altitude for hovering
        self.max_vel = 3.0   # Maximum vertical velocity limit

        # Proportional controller gain for altitude
        self.kp_z = 0.5

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        # Updates the current altitude from the pose message
        self.current_position.z = msg.pose.position.z

    def compute_velocity(self):
        # Calculate the error between target altitude and current altitude
        dz = self.altitude - self.current_position.z

        # Create the velocity command
        twist = Twist()
        twist.linear.x = 0  # No horizontal movement
        twist.linear.y = 0
        
        # Simple P-Controller for vertical velocity
        # The output is clamped between -max_vel and +max_vel to ensure safety
        twist.linear.z = max(min(dz * self.kp_z, self.max_vel), -self.max_vel)
        
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0 
        return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()

        # Wait for the flight controller to establish a connection
        while not rospy.is_shutdown() and (self.current_state is None or not self.current_state.connected):
            rate.sleep()

        while not rospy.is_shutdown():
            # Calculate and publish the velocity command continuously
            twist = self.compute_velocity()
            self.vel_pub.publish(twist)
            rate.sleep()

if __name__ == "__main__":
    controller = HoveringVelocity()
    controller.run()
