import rospy
from geometry_msgs.msg import Twist
from pidrone_pkg.msg import State
from sensor_msgs.msg import Range
from three_dim_vec import Position

class SurveillanceMode(object):
    def __init__(self):
        rospy.init_node("surveillance_node")

        # Publishers
        ############
        self.velpub = rospy.Publisher("/pidrone/surveilllance_velocity", Twist, queue_size=1)

        # Subscribers
        #############
        rospy.Subscriber("/pidrone/state", State, self.current_state_callback, queue_size=1)
        rospy.Subscriber('/pidrone/range', Range, self.altitude_callback, queue_size=1)

        # initialize the current state
        self.current_state = State()

        # Initialize the current position
        self.current_position = Position()

        # desired altitude
        self.desired_altitude = 1.0

        # desired route
        self.desired_route = [(2.0, 2.0), # dx (forward / backward), dy (right / left)
                              (4.0, 3.0),
                              (6.0, 5.0)]

        # index of current target
        self.current_target_index = 0

        # tolerance to reach target
        self.tolerance = 0.2

        # multiplicators for velocities
        self.kp_x = 0.2 # in s
        self.kp_y = 0.2
        self.kp_z = 0.2

        # parameter to avoid AttributeError
        self.altitude = 0.0

    def current_state_callback(self, state):
        """ Store the drone's current state for calculations """
        self.current_state = state
        pose = self.current_state.pose_with_covariance.pose
        self.current_position.x = pose.position.x
        self.current_position.y = pose.position.y
        self.current_position.z = pose.position.z

    def altitude_callback(self, msg):
        """
        The altitude of the robot
        Args:
            msg:  the message publishing the altitude

        """
        self.altitude = msg.range

    def velocity_calculator(self):
        if self.current_target_index >= len(self.desired_route):
            self.current_target_index = 0 # Reset to the first target if all targets are reached
        
        target = self.desired_route[self.current_target_index]
        # calculate distance to target
        dy = target[0] - self.current_position.x # translating forward is y, translating right is x
        dx = target[1] - self.current_position.y
        dz = self.desired_altitude - self.altitude
        distance = (dx**2 + dy**2) ** 0.5 # Pythagorean theorem

        if distance < self.tolerance:
            self.current_target_index += 1 # If too near at the current target, move to the next target without publishing velocity
            return
        
        # final velocities
        vx = dx * self.kp_x
        vy = dy * self.kp_y
        vz = dz * self.kp_z

        # set velocities
        twistMsg = Twist()
        twistMsg.linear.x = vx
        twistMsg.linear.y = vy
        twistMsg.linear.z = vz
        twistMsg.angular.x = 0
        twistMsg.angular.y = 0
        twistMsg.angular.z = 0
        self.velpub.publish(twistMsg)

if __name__ == '__main__':
    node = SurveillanceMode()
    rate = rospy.Rate(30)  # in Hz
    while not rospy.is_shutdown():
        node.velocity_calculator()
        rate.sleep()