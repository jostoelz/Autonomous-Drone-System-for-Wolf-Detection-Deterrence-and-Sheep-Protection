#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.normalized_pub = rospy.Publisher("/normalized_bounding_box", Point, queue_size=1)
        self.bool_pub = rospy.Publisher("/object_detector", Bool, queue_size=1)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_detected = False
            for c in contours:
                area = cv2.contourArea(c)
                # Schwellenwert auf 50 gesenkt!
                if area > 50:
                    object_detected = True
                    x, y, w, h = cv2.boundingRect(c)
                    height, width, _ = frame.shape
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    point_msg = Point()
                    point_msg.x = (center_x - width/2) / (width/2)
                    point_msg.y = (height/2 - center_y) / (height/2)
                    point_msg.z = 0
                    self.normalized_pub.publish(point_msg)
            
            self.bool_pub.publish(object_detected)
            
        except Exception as e:
            rospy.logerr(f"Fehler im Object Detector: {e}")

if __name__ == "__main__":
    try:
        ObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
