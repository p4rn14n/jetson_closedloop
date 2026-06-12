from pymata4 import pymata4
import time
import threading

board = pymata4.Pymata4()
ledreward = 12
board.set_pin_mode_digital_output(ledreward)

drop = int(input("Enter the number of drops:"))

def blink_led_pymata(led, t):
    board.digital_write(led,1)
    time.sleep(t)
    board.digital_write(led, 0)



for i in range(0, drop):
    t2 = threading.Thread(target=blink_led_pymata, args=(ledreward, 0.12))
    t2.start()
    time.sleep(1)