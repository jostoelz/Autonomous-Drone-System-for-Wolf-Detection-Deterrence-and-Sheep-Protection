#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Point, PoseStamped

class SurveillanceVelocity:
    def __init__(self):
        # Node
        rospy.init_node("surveillance_node")

        # Publisher für Velocity
        self.vel_pub = rospy.Publisher(
            "/surveillance_velocity", Twist, queue_size=10
        )

        # Subscriber für State
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        # Route
        self.route = [(0, 0), (10, 0), (10,10), (0, 10)]
        self.current_target_index = 0

        self.current_position = Point()
        self.current_position.x = 0.0
        self.current_position.y = 0.0
        self.current_position.z = 0.0
        self.altitude = 2.0

        self.max_vel = 3.0

        #  Controller Gains
        self.kp_x = 0.1
        self.kp_y = 0.1
        self.kp_z = 0.5
        self.tolerance = 0.5

    def state_cb(self, msg):
        self.current_state = msg
        # Optional: aktuelle Position aus MAVROS lesen
        # z.B. von /mavros/local_position/pose, sonst Annahme: selbstgeschätzt

    def pose_cb(self, msg):
        self.current_position.x = msg.pose.position.x
        self.current_position.y = msg.pose.position.y
        self.current_position.z = msg.pose.position.z

    def compute_velocity(self):
        target = self.route[self.current_target_index]
        dx = target[0] - self.current_position.x
        dy = target[1] - self.current_position.y
        rospy.loginfo_throttle(1.0, f"current_z={self.current_position.z:.3f}, target_z={self.altitude:.3f}")
        dz = self.altitude - self.current_position.z
        rospy.loginfo_throttle(1.0, f"dz={dz:.3f}")
        distance = (dx**2 + dy**2)**0.5

        if distance < self.tolerance:
            self.current_target_index += 1
            if self.current_target_index >= len(self.route):
                self.current_target_index = 0

        # PID control for height
        # if not hasattr(self, 'z_error_sum'):
            # self.z_error_sum = 0.0
            # self.z_error_prev = 0.0
        kp_z = 0.9
        ki_z = 0.05
        kd_z = 0.2
        # error
        # self.z_error_sum += dz
        # d_error = dz - self.z_error_prev
        # self.z_error_prev = dz
        # z_control = kp_z * dz + ki_z * self.z_error_sum + kd_z * d_error
        z_control = kp_z * dz
        z_control = max(min(z_control, self.max_vel), -self.max_vel)


        twist = Twist()
        twist.linear.x = max(min(dx * self.kp_x, self.max_vel), -self.max_vel)
        twist.linear.y = max(min(dy * self.kp_y, self.max_vel), -self.max_vel)
        twist.linear.z = z_control
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()

        # Warte auf Verbindung zum FCU
        while not rospy.is_shutdown() and (self.current_state is None or not self.current_state.connected):
            rate.sleep()

        rospy.loginfo("Verbindung hergestellt.")

        while not rospy.is_shutdown():
            # Velocity berechnen und senden
            twist = self.compute_velocity()
            self.vel_pub.publish(twist)
            rate.sleep()

if __name__ == "__main__":
    controller = SurveillanceVelocity()
    controller.run()
