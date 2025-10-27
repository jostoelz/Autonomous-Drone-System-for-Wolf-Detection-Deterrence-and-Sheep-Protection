#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

class DepthDetector:
    def __init__(self):
        rospy.init_node('depth_detector_node')
        self.bridge = CvBridge()

        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_cb)
        self.non_normalized_bounding_box_sub = rospy.Subscriber("/non_normalized_bounding_box", Point, self.non_normalized_bounding_box_cb)
        self.non_normalized_bounding_box = Point()
        self.depth_pub = rospy.Publisher("/depth", Float32, queue_size=10)

        self.latest_depth_frame = None

    def depth_image_cb(self, msg):
        # ROS-Image → OpenCV-Image konvertieren
        self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        if self.non_normalized_bounding_box is not None:
            self.depth_extraction()

    def non_normalized_bounding_box_cb(self, msg):
        self.non_normalized_bounding_box = msg

    def depth_extraction(self):
        if self.latest_depth_frame is None:
            return
        depth_image = self.latest_depth_frame
        height, width = depth_image.shape

        # Begrenzung, falls u/v außerhalb des Bildes liegen
        u = int(np.clip(self.non_normalized_bounding_box.x, 0, width - 1))
        v = int(np.clip(self.non_normalized_bounding_box.y, 0, height - 1))

        # Depth in Meter
        depth = float(depth_image[v, u])

        if np.isnan(depth) or np.isinf(depth):
            return

        # Publish
        depth_msg = Float32()
        depth_msg.data = depth
        self.depth_pub.publish(depth_msg)

if __name__ == "__main__":
    try:
        DepthDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

