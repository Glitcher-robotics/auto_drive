#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, rospkg
import numpy as np
import cv2, random, math
from cv_bridge import CvBridge #ready for opencv
from xycar_motor.msg import xycar_motor # ready for motor
from sensor_msgs.msg import Image #ready for camera

import sys
import os
import signal

def signal_handler(sig, frame): 
    os.system('killall -9 python rosout') #close program by insert ctrl+C
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#ready for transmit camera image data to opencv
image=np.empty(shape=[0])
bridge=CvBridge()
pub=None
Width=640
Height=480
Offset=340
Gap=40 #the image size is 640x480

#if camera topic is taken, call back the 'callback' (give image to opencv)
def img_callback(data):
    global image
    image=bridge.imgmsg_to_cv2(data, "bgr8")

def drive(Angle, Speed):
    global pub

    msg=xycar_motor()
    msg.angle=Angle
    msg.speed=Speed #receive Angle, Speed

    pub.publish(msg) #publish xycar_motor topic to motor node

#draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #draw lines with random color above the image
        img=cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)
    return img

#draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0): #draw small rectangle on the image
    center=(lpos+rpos)/2

    cv2.rectangle(img, (lpos-5, 15+offset), (lpos+5, 25+offset), (0, 255, 0), 2) #draw green rectangle at the lpos
    cv2.rectangle(img, (rpos-5, 15+offset), (rpos+5, 25+offset), (0, 255, 0), 2) #draw at rpos
    cv2.rectangle(img, (center-5, 15+offset), (center+5, 25+offset), (0, 255, 0), 2) #draw at center of the r&l rectangle
    cv2.rectangle(img, (315, 15+offset), (325, 25+offset), (0, 0, 255), 2)#draw red rectangle at center
    return img

# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 10 # distinguish extracted line to r&l line on the basis of r&l

    #calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2-x1==0:
            slope = 0 # find line's slope and extract line whose slope is less than 10
        else:
            slope = float(y2-y1) / float(x2-x1)
        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])
        
        #divide lines left to right
        left_lines = []
        right_lines = [] # the calculation method is different, because y is increse to downside in opencv coordinates

        for j in range(len(slopes)):
            Line = new_lines[j]
            slope = slopes[j]
            x1, y1, x2, y2 = Line 
            if (slope<0) and (x2 < Width/2 - 90):
                left_lines.append([Line.tolist()]) # Gathering negative slope line in leftside
            elif (slope > 0) and (x1 > Width/2 + 90):
                right_lines.append([Line.tolist()]) # Gathering positive slope line in rightside
            
        return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2 # find mean value of m&b at Parameter space targeting the lines we found
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)

    if m == 0 and b == 0: # if line is not detected, set the left to 0 and right to width(640) (edge of the image)
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m # to sign the track line on the image find the 0 point by add the offset value(340) to y-intercept

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m) # draw the track line on the image by find the x1 and x2 value in the image downside and center side

        cv2.line(img, (int(x1), Height), (int(x2), (Height/2)), (255, 0, 0), 3)

    return img, int(pos)

# show image and return lpos, rpos, camera image processing

def process_image(frame):
    global Width
    global Offset, Gap

    #gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # canny edge
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    # HoughLinesP
    roi = edge_img[Offset : Offset+Gap, 0 : Width]
    all_lines = cv2.HoughLinesP(roi, 1, math.pi/180, 30, 30, 10)

    # divide left, right lines
    if all_lines is None:
        return 0, 640
    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True) # receive line information and draw line track on the image and find pose

    # draw lines with random color in the ROI by Hough Transform
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255, 255, 255), 2)

    # draw rectangle on the line and image center
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)

    # show image
    cv2.imshow('calibration', frame)
    return lpos, rpos

def start():

    global pub
    global image
    global cap
    global Width, Height

    rospy.init_node('auto_drive') # init auto_drive node
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # publish xycar_motor topic

    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, img_callback) #subscribe usb_cam topic to OpenCV

    print "----------Xycar A2 KM v1.0 ----------"
    rospy.sleep(2)

    while not image.size == (640*480*3):
        continue # receive camgera image data

    while True:

        lpos, rpos = process_image(image) # find left line pose and right line pose, image processing based on hough transform

        center = (lpos + rpos) / 2
        angle = (Width/2 - center) # find angel with left and right side line and center point and image center, decide angle and publish xycar_motor topic to motor

        drive(angle, 20) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    rospy.spin()

if __name__ == '__main__':

    start()