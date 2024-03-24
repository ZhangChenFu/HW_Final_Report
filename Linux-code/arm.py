import time
import hiwonder

jetmax = hiwonder.JetMax()
sucker = hiwonder.Sucker()

if __name__ == '__main__':
    jetmax.go_home()
    time.sleep(4)
    hiwonder.pwm_servo1.set_position(90 , 0.1)
    time.sleep(2)
    cur_x, cur_y, cur_z = jetmax.position
    #0 -162.94 212.8
    print(cur_x, cur_y, cur_z)
    
    while True:
        # left up	x,80	y,80
        #jetmax.set_position((cur_x+95, cur_y-100, cur_z-105), 1)
        #time.sleep(4)
        jetmax.set_position((cur_x, cur_y, cur_z), 1)
        time.sleep(4)

        # right up	x,550	y,80
        #jetmax.set_position((cur_x-95, cur_y-100, cur_z-105), 1)
        #time.sleep(4)
        #jetmax.set_position((cur_x, cur_y, cur_z), 1)
        #time.sleep(4)
        
        # left down	x,80	y,400
        #jetmax.set_position((cur_x+95, cur_y, cur_z-105), 1)
        #time.sleep(4)
        #jetmax.set_position((cur_x, cur_y, cur_z), 1)
        #time.sleep(4)
        
        # right down	x,570	y,400
        #jetmax.set_position((cur_x-100, cur_y, cur_z-105), 1)
        #time.sleep(4)
        #jetmax.set_position((cur_x, cur_y, cur_z), 1)
        #time.sleep(4)

        # x,2.5each	y,3.2each

        # center	x,320	y,240
        jetmax.set_position((cur_x-10, cur_y-50, cur_z-159), 1)
        time.sleep(4)
        #sucker.set_state(True)
        #time.sleep(2)
        jetmax.set_position((cur_x+150, cur_y+100, cur_z), 1)
        time.sleep(4)
        #sucker.release(2)
        #time.sleep(4)


