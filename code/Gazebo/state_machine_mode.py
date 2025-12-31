#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool

class StateMachine:
    def __init__(self):
        rospy.init_node("state_machine_node")
        
        # Publisher for the final velocity command sent to the flight controller
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)

        # Subscribers for system state and input velocity streams
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

        # Service clients for arming and mode switching
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # Internal state variables
        self.last_object_seen_time = rospy.Time.now()
        self.hover_duration = rospy.Duration(3.0)
        self.hovering = False

    # --- Callback Functions ---
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
        """
        Determines the appropriate velocity command based on the current
        operational state (Surveillance, Chasing, or Hovering).
        """
        twist = Twist()
        now = rospy.Time.now()

        # The following logic handles automatic state transitions
        '''
        if self.object_detected.data:
            # CHASING MODE
            twist = self.chasing_velocity
            self.last_object_seen_time = now
            self.hovering = False
            rospy.loginfo_throttle(1, "State: Chasing")
        else:
            time_since_seen = now - self.last_object_seen_time
            if time_since_seen < self.hover_duration:
                # HOVERING MODE (Target lost recently)
                twist = self.hovering_velocity
                self.hovering = True
                rospy.loginfo_throttle(1, "State: Hovering")
            else:
                # SURVEILLANCE MODE
                twist.linear.x = self.surveillance_velocity.linear.x
                
                # --- Correction: Enforce zero lateral velocity ---
                # twist.linear.y = 0.0 
                # -------------------------------------------------
                
                twist.linear.z = self.surveillance_velocity.linear.z
                twist.angular.z = self.surveillance_velocity.angular.z
                self.hovering = False
                rospy.loginfo_throttle(1, "State: Surveillance (Y-axis restricted)")
        '''
        
        # Default pass-through (Overridden logic for testing)
        twist = self.surveillance_velocity

        return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()
        
        # Wait for the flight controller to connect
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()

        # Main control loop
        while not rospy.is_shutdown():
            # Periodically attempt to switch to OFFBOARD mode if not active
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.set_mode_client(0, "OFFBOARD")
                last_req = rospy.Time.now()
            
            # Periodically attempt to ARM the vehicle if not armed
            elif not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.arming_client(True)
                last_req = rospy.Time.now()

            twist = self.decision_maker()
            self.vel_pub.publish(twist)
            rate.sleep()

if __name__ == "__main__":
    StateMachine().run()
