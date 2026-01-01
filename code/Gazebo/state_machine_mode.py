#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

class StateMachine:
    def __init__(self):
        rospy.init_node("state_machine_node")
        
        # Publisher: Sends velocity commands to the flight controller
        # Uses TwistStamped to include header information (frame_id and timestamp)
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)

        # --- Subscribers ---
        
        # 1. State: Monitors the connection status, arming state, and flight mode
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()
        
        # 2. Pose: Essential for determining the drone's current heading (yaw)
        # This is required to transform local velocity commands into the global frame
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)
        self.current_yaw = 0.0

        # 3. Input Channels: Receives velocity commands from different control behaviors
        rospy.Subscriber("/surveillance_velocity", Twist, self.surveillance_cb)
        self.surveillance_velocity = Twist()
        
        rospy.Subscriber("/chasing_velocity", Twist, self.chasing_cb)
        self.chasing_velocity = Twist()
        
        rospy.Subscriber("/object_detector", Bool, self.object_detector_cb)
        self.object_detected = Bool()
        
        rospy.Subscriber("/hovering_velocity", Twist, self.hovering_cb)
        self.hovering_velocity = Twist()

        # Service Clients: Required to switch modes and arm the vehicle programmatically
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

    def pose_cb(self, msg):
        # Extracts the current orientation of the drone
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        # Converts Quaternion to Euler angles to isolate the Yaw (heading)
        # This yaw value is critical for coordinate frame transformation
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_yaw = yaw

    def surveillance_cb(self, msg):
        self.surveillance_velocity = msg

    def chasing_cb(self, msg):
        self.chasing_velocity = msg

    def object_detector_cb(self, msg):
        self.object_detected = msg

    def hovering_cb(self, msg):
        self.hovering_velocity = msg

    def decision_maker(self):
        # Logic to select the active velocity source
        # Currently forces the 'Chasing' mode for testing purposes
        twist = Twist()
        twist = self.chasing_velocity 
        return twist

    def run(self):
        rate = rospy.Rate(30)
        last_req = rospy.Time.now()
        
        # Wait for the flight controller to establish a connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()

        rospy.loginfo("StateMachine started. Performing manual Body-to-Global frame transformation...")

        while not rospy.is_shutdown():
            # Offboard and Arming Logic
            # Attempts to switch to OFFBOARD mode and ARM the drone at 5-second intervals if not already active
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.set_mode_client(0, "OFFBOARD")
                last_req = rospy.Time.now()
            elif not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.arming_client(True)
                last_req = rospy.Time.now()

            # 1. Retrieve Input: Gets the desired velocity relative to the drone body
            # (e.g., "1 m/s forward")
            raw_input = self.decision_maker()
            
            # 2. Coordinate Transformation: Body Frame -> Global Frame (Map)
            # Assumes raw_input.linear.x represents "Forward"
            # Assumes raw_input.linear.y represents "Sideways" (usually 0 in this context)
            
            desired_forward = raw_input.linear.x
            desired_side    = raw_input.linear.y
            
            # Application of a 2D Rotation Matrix
            # Global X = Forward * cos(yaw) - Side * sin(yaw)
            # Global Y = Forward * sin(yaw) + Side * cos(yaw)
            
            global_vel_x = (desired_forward * math.cos(self.current_yaw)) - (desired_side * math.sin(self.current_yaw))
            global_vel_y = (desired_forward * math.sin(self.current_yaw)) + (desired_side * math.cos(self.current_yaw))
            
            # 3. Message Construction
            final_cmd = TwistStamped()
            final_cmd.header.stamp = rospy.Time.now()
            final_cmd.header.frame_id = "map"  # Explicitly states that coordinates are Global/ENU
            
            final_cmd.twist.linear.x = global_vel_x
            final_cmd.twist.linear.y = global_vel_y
            final_cmd.twist.linear.z = raw_input.linear.z
            
            # Angular velocity (Yaw Rate) remains unchanged across frames
            # A rotation speed of 0.5 rad/s is the same in Body and Global frames
            final_cmd.twist.angular.z = raw_input.angular.z

            self.vel_pub.publish(final_cmd)
            rate.sleep()

if __name__ == "__main__":
    StateMachine().run()
