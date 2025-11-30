from gpiozero import PWMOutputDevice
from time import sleep

# We use PWMOutputDevice because it is more flexible for raw frequencies
# Pin 17 is the GPIO pin (physical pin 11)
buzzer = PWMOutputDevice(17)

print("Press CTRL+C to exit")

try:
    while True:
        # 1. Set frequency 
        buzzer.frequency = 4000
        
        # 2. Turn sound on
        # This means: 50% of the time on, 50% off -> This generates the sound wave.
        buzzer.value = 0.5
        
        sleep(1.0)
        
        # 3. Turn sound off (0.0 = 0%)
        buzzer.value = 0
        
        sleep(1.5)

except KeyboardInterrupt:
    print("Program ended")
    buzzer.off()
