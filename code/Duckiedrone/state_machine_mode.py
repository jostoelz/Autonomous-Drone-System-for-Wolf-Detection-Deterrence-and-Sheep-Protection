import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D

class StateMachineMode(object):
    def __init__(self):
        rospy.init_node("state_machine_node")

        # Publishers
        ############
        self.velpub = rospy.Publisher("/pidrone/final/twist", Twist, queue_size=1)

        # Subscribers
        #############
        rospy.Subscriber('/pidrone/chasing_velocity', Twist, self.chasing_velocity_callback, queue_size=1)
        rospy.Subscriber('/pidrone/surveillance_velocity', Twist, self.surveillance_velocity_callback, queue_size=1)
        rospy.Subscriber('/pidrone/desired/twist', Twist, self.keyboard_velocity_callback, queue_size=1)
        rospy.Subscriber('/pidrone/manuel_control', Bool, self.manuel_control_callback, queue_size=1)
        rospy.Subscriber('/pidrone/object_detections', Detection2D, self.bounding_box_callback, queue_size=1)
        
        # parameters to avoid AttributeError
        self.chasing_vx = 0.0
        self.chasing_vy = 0.0
        self.chasing_vz = 0.0
        self.surveillance_vx = 0.0
        self.surveillance_vy = 0.0
        self.surveillance_vz = 0.0
        self.keyboard_vx = 0.0
        self.keyboard_vy = 0.0
        self.keyboard_vz = 0.0
        self.manuel_control = False
        self.horizontal_shift = 0.0
        self.vertical_shift = 0.0

        # image dimensions
        self.image_width = 320
        self.image_height = 240

    def chasing_velocity_callback(self, msg):
        self.chasing_vx = msg.linear.x
        self.chasing_vy = msg.linear.y
        self.chasing_vz = msg.linear.z

    def surveillance_velocity_callback(self, msg):
        self.surveillance_vx = msg.linear.x
        self.surveillance_vy = msg.linear.y
        self.surveillance_vz = msg.linear.z
    
    def keyboard_velocity_callback(self, msg):
        self.keyboard_vx = msg.linear.x
        self.keyboard_vy = msg.linear.y
        self.keyboard_vz = msg.linear.z

    def manuel_control_callback(self, msg):
        self.manuel_control = msg.data
    
    def bounding_box_callback(self, msg):
        if msg.bbox.size_x > 0 and msg.bbox.size_y > 0: # check if there is a detection, else set shifts to 0
            self.horizontal_shift = (msg.bbox.center.x - self.image_width/2) / (self.image_width/2) # normalized between -1 and 1, 0 is center
            self.vertical_shift = (msg.bbox.center.y - self.image_height/2) / (self.image_height/2)
        else:
            self.horizontal_shift = 0.0
            self.vertical_shift = 0.0

    def decision_maker(self):
        twistMsg = Twist()

        if self.manuel_control:
            twistMsg.linear.x = self.keyboard_vx
            twistMsg.linear.y = self.keyboard_vy
            twistMsg.linear.z = self.keyboard_vz
        elif self.horizontal_shift != 0.0 or self.vertical_shift != 0.0: # check if there is an object detected
            twistMsg.linear.x = self.chasing_vx
            twistMsg.linear.y = self.chasing_vy
            twistMsg.linear.z = self.chasing_vz
        else:
            twistMsg.linear.x = self.surveillance_vx
            twistMsg.linear.y = self.surveillance_vy
            twistMsg.linear.z = self.surveillance_vz

        self.velpub.publish(twistMsg)

if __name__ == '__main__':
    node = StateMachineMode()
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        node.decision_maker()
        rate.sleep()
