from pymata4 import pymata4

ledLightTTL = 13

board = pymata4.Pymata4()

board.set_pin_mode_digital_output(ledLightTTL)

board.digital_write(ledLightTTL,0)