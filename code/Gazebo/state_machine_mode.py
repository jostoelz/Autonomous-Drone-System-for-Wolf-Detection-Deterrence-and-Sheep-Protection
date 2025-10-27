#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Bool

class StateMachine:
    def __init__(self):
        # Node
        rospy.init_node("state_machine_node")

        # Publisher
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)

        # Subscriber
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()
        rospy.Subscriber("/surveillance_velocity", Twist, self.surveillance_cb)
        self.surveillance_velocity = Twist()
        rospy.Subscriber("/chasing_velocity", Twist, self.chasing_cb)
        self.chasing_velocity = Twist()
        rospy.Subscriber("/object_detector", Bool, self.object_detector_cb)
        self.object_detected = Bool()
        rospy.Subscriber("/hovering_velocity", Twist, self.hovering_cb)
        self.hovering_velocity = Twist()

        # OFFBOARD & Arming Setup
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.last_object_seen_time = rospy.Time.now()
        self.hover_duration = rospy.Duration(3.0)
        self.hovering = False

    def state_cb(self, msg):
        self.current_state = msg

    def surveillance_cb(self, msg):
        self.surveillance_velocity = msg

    def chasing_cb(self, msg):
        self.chasing_velocity = msg

    def object_detector_cb(self, msg):
        self.object_detected = msg

    def hovering_cb(self, msg):
        self.hovering_velocity = msg

    def decision_maker(self):
        twist = Twist()
        now = rospy.Time.now()

        if self.object_detected.data:
            twist.linear.x = self.chasing_velocity.linear.x
            twist.linear.y = self.chasing_velocity.linear.y
            twist.linear.z = self.chasing_velocity.linear.z
            twist.angular.x = self.chasing_velocity.angular.x
            twist.angular.y = self.chasing_velocity.angular.y
            twist.angular.z = self.chasing_velocity.angular.z
            self.last_object_seen_time = now
            self.hovering = False
            rospy.loginfo("chasing modus")
        else:
            time_since_seen = now - self.last_object_seen_time

            if time_since_seen < self.hover_duration:
                twist.linear.x = self.hovering_velocity.linear.x
                twist.linear.y = self.hovering_velocity.linear.y
                twist.linear.z = self.hovering_velocity.linear.z
                twist.angular.x = self.hovering_velocity.angular.x
                twist.angular.y = self.hovering_velocity.angular.y
                twist.angular.z = self.hovering_velocity.angular.z
                self.hovering = True
                rospy.loginfo("hovering modus")

            else:
                twist.linear.x = self.surveillance_velocity.linear.x
                twist.linear.y = self.surveillance_velocity.linear.y
                twist.linear.z = self.surveillance_velocity.linear.z
                twist.angular.x = self.surveillance_velocity.angular.x
                twist.angular.y = self.surveillance_velocity.angular.y
                twist.angular.z = self.surveillance_velocity.angular.z
                self.hovering = False
                rospy.loginfo("surveillance modus")

        return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()

        # Warte auf Verbindung zum FCU
        while not rospy.is_shutdown() and (self.current_state is None or not self.current_state.connected):
            rate.sleep()

        rospy.loginfo("Verbindung hergestellt. Starte OFFBOARD und Arming...")

        while not rospy.is_shutdown():
            # OFFBOARD setzen
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.set_mode_client(0, "OFFBOARD")
                last_req = rospy.Time.now()
                rospy.loginfo("OFFBOARD gesetzt")

            # Arming
            elif not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.arming_client(True)
                last_req = rospy.Time.now()
                rospy.loginfo("Drohne ge-armed")

            # Velocity berechnen und senden
            twist = self.decision_maker()
            self.vel_pub.publish(twist)
            rate.sleep()

if __name__ == "__main__":
    controller = StateMachine()
    controller.run()
