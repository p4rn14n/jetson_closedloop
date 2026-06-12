from pymata4 import pymata4
import time
import threading

board = pymata4.Pymata4()
ledreward = 12
board.set_pin_mode_digital_output(ledreward)

seconds = int(input("how long (seconds)?:"))

def blink_led_pymata(led, t):
    board.digital_write(led,1)
    time.sleep(t)
    board.digital_write(led, 0)




t2 = threading.Thread(target=blink_led_pymata, args=(ledreward, seconds))
t2.start()
    # time.sleep(1)