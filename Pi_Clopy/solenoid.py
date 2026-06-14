from gpiozero import LED
from threading import *
import time

def blink_led(led, t):
    led.on()
    time.sleep(t)
    led.off()
    
    
ledReward = LED(27)

x = int(input("how long?"))

t2 = Thread(target=blink_led, args=(ledReward, x))
t2.start()
