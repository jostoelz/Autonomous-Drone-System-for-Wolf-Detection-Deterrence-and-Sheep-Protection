import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from vision_msgs.msg import Detection2D
from std_msgs.msg import Float32

class ChasingMode(object):
    def __init__(self):
        rospy.init_node("chasing_node")

        # Publishers
        ############
        self.velpub = rospy.Publisher("/pidrone/chasing_velocity", Twist, queue_size=1)

        # Subscribers
        #############
        rospy.Subscriber('/pidrone/range', Range, self.altitude_callback, queue_size=1)
        rospy.Subscriber('/pidrone/object_depth', Float32, self.depth_callback, queue_size=1)
        rospy.Subscriber('/pidrone/object_detections', Detection2D, self.bounding_box_callback, queue_size=1)
        
        # minimum altitude
        self.min_altitude = 1.0

        # minimum distance to object
        self.min_distance_object = 1.0

        # multiplicators for velocities
        self.kp_x = 0.5
        self.kp_y = 0.5
        self.kp_z = 1.0

        # parameters to avoid AttributeError
        self.altitude = 0.0
        self.current_distance = 0.0
        self.horizontal_shift = 0.0
        self.vertical_shift = 0.0

        # image dimensions
        self.image_width = 320
        self.image_height = 240
    def altitude_callback(self, msg):
        """
        The altitude of the robot
        Args:
            msg:  the message publishing the altitude

        """
        self.altitude = msg.range

    def depth_callback(self, msg):
        self.current_distance = msg.data
    
    def bounding_box_callback(self, msg):
        self.horizontal_shift = (msg.bbox.center.x - self.image_width/2) / (self.image_width/2) # normalized between -1 and 1, 0 is center
        self.vertical_shift = (msg.bbox.center.y - self.image_height/2) / (self.image_height/2)

    def velocity_calculator(self):
        # calculate distance to target
        dy = self.current_distance - self.min_distance_object # translating forward is y, translating right is x
        dx = self.horizontal_shift

        if self.altitude > self.min_altitude: # if drone is too low, set minimum altitude
            dz = self.vertical_shift
        else:
            dz = self.min_altitude - self.altitude
        
        # final velocities
        vx = dx * self.kp_x
        vy = dy * self.kp_y
        vz = dz * self.kp_z

        # set velocities
        twistMsg = Twist()
        twistMsg.linear.x = vx
        twistMsg.linear.y = vy
        twistMsg.linear.z = vz
        self.velpub.publish(twistMsg)

    if __name__ == '__main__':
        node = ChasingMode()
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            node.velocity_calculator()
            rate.sleep()