#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
from gpiozero import PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

# --- Configuration ---
PIN_NUMBER = 24
FREQUENCY = 4000  # 4 kHz sound frequency
TOPIC_NAME = "chasing_mode"

# --- Hardware Setup ---
# Connects to the pigpio daemon running on the host
factory = PiGPIOFactory()
# Initializes the buzzer but keep it off (initial_value=0) start
buzzer = PWMOutputDevice(PIN_NUMBER, pin_factory=factory, frequency=FREQUENCY, initial_value=0)

def mode_callback(msg):
    """
    Callback triggered whenever a message arrives on the topic.
    msg.data contains the boolean value.
    """
    if msg.data:
        # Chasing mode is True -> Turn sound ON
        # 0.5 means 50% duty cycle, which is usually max volume for a buzzer
        rospy.loginfo("Chasing mode ON: Playing sound.")
        buzzer.value = 0.5
    else:
        # Chasing mode is False -> Turn sound OFF
        rospy.loginfo("Chasing mode OFF: Silence.")
        buzzer.value = 0.0
def cleanup():
    """Run this when the node is killed to stop the noise."""
    buzzer.off()
    buzzer.close()

if __name__ == '__main__':
    try:
        # Init the ROS node
        rospy.init_node('chasing_buzzer_controller')

        # Register shutdown hook
        rospy.on_shutdown(cleanup)

        # Subscribe to the topic
        rospy.Subscriber(TOPIC_NAME, Bool, mode_callback)

        # Keep the node alive
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

