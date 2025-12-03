from gpiozero import PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

# connects to hardware deamon
factory = PiGPIOFactory()

buzzer = PWMOutputDevice(24, pin_factory=factory)

try:
    # half of the time flows current  
    buzzer.value = 0.5
    
    # set frequency to 4 kHz 
    buzzer.frequency = 4000
    
    # infinity loop so that file doesn't stop 
    while True:
        sleep(1)

except KeyboardInterrupt:
    buzzer.off()
