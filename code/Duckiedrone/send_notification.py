#!/usr/bin/env python
import rospy
import requests
from std_msgs.msg import Bool

# --- Configuration ---
TOPIC_NAME = "chasing_mode"
NTFY_TOPIC = "wolf_detection"  # Your ntfy.sh topic name

# State variable to prevent spamming notifications
# Only sends one notification when the mode switches to True
notification_sent = False

def send_text_message(message):
    """Sends the actual HTTP POST request to ntfy.sh"""
    try:
        requests.post(
            "https://ntfy.sh/" + NTFY_TOPIC,
            data=message.encode('utf-8'),
            headers={
                "Title": "ALARM", 
                "Priority": "high", 
                "Tags": "warning, wolf" 
            }
        )
    except Exception as e:
        rospy.logerr("Failed to send notification: " + str(e))

def callback(msg):
    global notification_sent
    
    # Check if Chasing Mode is active (True)
    if msg.data: 
        # Only send if we haven't sent it already in this cycle
        if not notification_sent:
            send_text_message("WARNING: Wolf detected! Check your sheep immediately!")
            notification_sent = True
    else:
        # Chasing Mode is OFF (False)
        # We reset the flag so we are ready to notify again next time it turns on
        if notification_sent:
            notification_sent = False

def listener():
    # Init Node
    rospy.init_node('notification_sender_node')
    
    # Subscribe
    rospy.Subscriber(TOPIC_NAME, Bool, callback)
    
    # Keep alive
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

