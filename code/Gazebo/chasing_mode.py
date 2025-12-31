#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Point, PoseStamped, Twist
from std_msgs.msg import Float32

class ChasingVelocity:
    def __init__(self):
        # Node
        rospy.init_node("chasing_node")

        # Publisher
        self.vel_pub = rospy.Publisher("/chasing_velocity", Twist, queue_size=10)

        # Subscriber
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()
        rospy.Subscriber("/normalized_bounding_box", Point, self.normalized_bounding_box_cb)
        self.normalized_bounding_box = Point()
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)
        self.current_position = Point()
        rospy.Subscriber("/depth", Float32, self.depth_cb)
        self.depth = Float32()
        self.altitude = 2.0
        self.min_distance_object = 1.0
        self.max_vel = 3.0
        self.max_yaw_rate = 2.0

        #  Controller Gains
        self.kp_x = 0.1
        self.kp_yaw = 0.5
        self.kp_z = 0.5

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_position.z = msg.pose.position.z

    def normalized_bounding_box_cb(self, msg):
        self.normalized_bounding_box = msg

    def depth_cb(self, msg):
        self.depth = msg

    def compute_velocity(self):
       dx = self.depth.data - self.min_distance_object
       dy = self.normalized_bounding_box.x
       dz = self.altitude - self.current_position.z
       twist = Twist()
       twist.linear.x = max(min(dx * self.kp_x, self.max_vel), -self.max_vel)
       twist.linear.y = 0
       twist.linear.z = max(min(dz * self.kp_z, self.max_vel), -self.max_vel)
       twist.angular.x = 0
       twist.angular.y = 0
       twist.angular.z = max(min(-dy * self.kp_yaw, self.max_yaw_rate), -self.max_yaw_rate)
       return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()

        # Warte auf Verbindung zum FCU
        while not rospy.is_shutdown() and (self.current_state is None or not self.current_state.connected):
            rate.sleep()

        rospy.loginfo("Verbindung hergestellt.")

        while not rospy.is_shutdown():
            twist = self.compute_velocity()
            self.vel_pub.publish(twist)
            rate.sleep()

if __name__ == "__main__":
    controller = ChasingVelocity()
    controller.run()
