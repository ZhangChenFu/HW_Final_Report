#!/usr/bin/env python3
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from jetmax_control.msg import SetServo
import hiwonder
import queue
import pupil_apriltags as apriltag
import yaml



ROS_NODE_NAME = "apriltag_detector"
TAG_SIZE = 33.30


class AprilTagDetect:
    def __init__(self):
        self.camera_params = None
        self.K = None
        self.R = None
        self.T = None


    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/block_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)


    #cv2.waitKey(1)


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass


if __name__ == '__main__':
    state = AprilTagDetect()
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    at_detector = apriltag.Detector()
    image_queue = queue.Queue(maxsize=1)
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    jetmax.go_home()
    state.load_camera_params()
    myimage = None
    if state.camera_params is None:
        rospy.logerr('Can not load camera parameters')
        sys.exit(-1)
    rospy.ServiceProxy('/jetmax/go_home', Empty)()
    rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1).publish(data=False)
    rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1).publish(data=90, duration=0.5)
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)
    while (True):
        ros_image = image_queue.get(block=True)
        image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        frame_result = image.copy()
        bgr_image = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)
        cv2.imshow('result', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            myimage = bgr_image
            break
    
    cv2.destroyAllWindows()
    cv2.imwrite(r'pictures/get.png',myimage)
    jetmax.go_home()
    time.sleep(4)
    hiwonder.pwm_servo1.set_position(90 , 0.1)
    time.sleep(2)
    cur_x, cur_y, cur_z = jetmax.position

    gray = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    reverse = 255 - gray
    ret1, binary_otsu = cv2.threshold(reverse, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_otsu, kernel, iterations=1)
    image_dilate = cv2.dilate(image_erosion, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(image_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    center = (x+(w//2), y+(h//2))
    cx, cy = center
    cx = int((320 - cx)/2.2)
    cy = int((cy-400)/3.0)
    print(cx, cy)

    jetmax.set_position((cur_x + cx, cur_y + cy, cur_z - 159), 1)
    time.sleep(4)
    sucker.set_state(True)
    time.sleep(2)
    jetmax.set_position((cur_x+150, cur_y+100, cur_z), 1)
    time.sleep(4)
    sucker.release(2)
    time.sleep(4)
    jetmax.go_home()
    time.sleep(4)

