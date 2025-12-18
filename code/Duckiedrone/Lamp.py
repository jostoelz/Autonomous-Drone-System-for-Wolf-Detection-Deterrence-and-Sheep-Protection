from gpiozero import PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

# --- Config ---
PIN_NUMBER = 25       
MAX_POWER = 0.80  # Limit to ~13.5V to protect hardware

# Use pigpio for stable, hardware-timed PWM
factory = PiGPIOFactory()
lamp = PWMOutputDevice(PIN_NUMBER, pin_factory=factory, frequency=7000)

try:
    # Ramp up brightness
    current_val = 0.0
    while current_val < MAX_POWER:
        current_val += 0.01
        if current_val > MAX_POWER:
            current_val = MAX_POWER

        lamp.value = current_val
        sleep(0.05)

    while True:
        sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Safe shutdown sequence
    lamp.value = 0
    lamp.close()
