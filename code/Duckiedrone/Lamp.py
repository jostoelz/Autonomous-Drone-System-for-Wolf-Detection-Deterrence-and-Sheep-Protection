#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
from gpiozero import PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

# --- Configuration ---
PIN_NUMBER = 25
MAX_POWER = 0.7  # Cap the output to protect hardware
TOPIC_NAME = "chasing_mode"

# --- Hardware Setup ---
# Note: 'sudo pigpiod' must be running in the background
factory = PiGPIOFactory()
lamp = PWMOutputDevice(PIN_NUMBER, pin_factory=factory, frequency=7000)

def mode_callback(msg):
    """
    Callback triggered whenever a message arrives on the topic.
    msg.data contains the boolean value.
    """
    if msg.data:
        # Chasing mode is True -> Turn light ON
        lamp.value = MAX_POWER
    else:
        # Chasing mode is False -> Turn light OFF
        lamp.value = 0.0
def cleanup():
    """Run this when the node is killed (Ctrl+C) to prevent the light getting stuck on."""
    lamp.value = 0.0
    lamp.close()

if __name__ == '__main__':
    try:
        # Init the ROS node
        rospy.init_node('chasing_light_controller')

        # Register the shutdown hook for clean exit
        rospy.on_shutdown(cleanup)

        # Subscribe to the topic, expected is a std_msgs/Bool message type
        rospy.Subscriber(TOPIC_NAME, Bool, mode_callback)

        # Keep the node alive to listen for messages
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
