import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import time

class PhotoTaker:
    def __init__(self):
        rospy.init_node('snapshot_taker', anonymous=True)
        self.image_saved = False
        self.subscriber = rospy.Subscriber("/raspicam_node/image/compress$

    def callback(self, msg):
        if not self.image_saved:
            try:
                # conversion of compressed image to openCV format
                np_arr = np.frombuffer(msg.data, np.uint8)
                image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # file name with time stamp
                filename = "duckie_photo_" + str(int(time.time())) + ".jp$
                cv2.imwrite(filename, image_np)

                rospy.loginfo("Photo saved as: " + filename)
                self.image_saved = True
                rospy.signal_shutdown("finished")
            except Exception as e:
                rospy.logerr("Error while saving: " + str(e))

if __name__ == '__main__':
    pt = PhotoTaker()
    rospy.spin()

