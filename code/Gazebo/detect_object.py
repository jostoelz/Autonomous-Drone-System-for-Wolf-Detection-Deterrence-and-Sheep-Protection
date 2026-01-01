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
        # Initialize the ROS node
        rospy.init_node('object_detector_node')
        
        # Bridge to convert between ROS Image messages and OpenCV images
        self.bridge = CvBridge()
        
        # Subscriber to the raw color camera stream
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        # Publisher 1: Normalized Coordinates
        # Sends relative position (-1 to 1) for the flight controller
        self.normalized_pub = rospy.Publisher("/normalized_bounding_box", Point, queue_size=1)
        
        # Publisher 2: Pixel Coordinates
        # Critical for the Depth Detector to map 2D center points to 3D depth data
        self.non_normalized_pub = rospy.Publisher("/non_normalized_bounding_box", Point, queue_size=1)
        
        # Publisher indicating if an object is currently visible
        self.bool_pub = rospy.Publisher("/object_detector", Bool, queue_size=1)

    def image_callback(self, msg):
        try:
            # Convert ROS image message to an OpenCV BGR image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert BGR to HSV color space for better color segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # --- COLOR CONFIGURATION ---            
            # Setting: BLUE (Active configuration)
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            

            # Create a binary mask where the detected color is white
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Morphological operations to remove noise (small specs) from the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Debug Visualization
            # Opens windows to show the processed mask and the camera feed
            cv2.imshow("Mask Debug", mask)
            cv2.imshow("Camera View", frame)
            cv2.waitKey(1)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_detected = False
            
            # Logic to select the LARGEST contour, ignoring small noise artifacts
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                
                # Threshold to ensure the object is significant enough to track
                if area > 100: 
                    object_detected = True
                    x, y, w, h = cv2.boundingRect(c)
                    height, width, _ = frame.shape
                    
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # 1. Publish Normalized Coordinates (Control Input)
                    # Maps the position to a range of -1 (Left/Top) to 1 (Right/Bottom)
                    norm_msg = Point()
                    norm_msg.x = (center_x - width/2) / (width/2) 
                    norm_msg.y = (height/2 - center_y) / (height/2) 
                    norm_msg.z = 0
                    self.normalized_pub.publish(norm_msg)

                    # 2. Publish Non-Normalized Coordinates (Depth Mapping)
                    # Sends raw pixel coordinates (u, v) for depth extraction
                    pixel_msg = Point()
                    pixel_msg.x = center_x # Pixel u
                    pixel_msg.y = center_y # Pixel v
                    pixel_msg.z = 0
                    self.non_normalized_pub.publish(pixel_msg)

                    # Draw a green bounding box on the frame for visualization
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Publish detection status
            self.bool_pub.publish(object_detected)
            
        except Exception as e:
            rospy.logerr(f"Error in Object Detector: {e}")

if __name__ == "__main__":
    try:
        ObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
