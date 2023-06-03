#!/usr/bin/env python3

import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
from multilane_sorter.msg import inference
import time
import numpy as np
from rembg import remove,new_session


class PreProcessing():
    def __init__(self):
        rospy.init_node('preprocessing_node')
        # rospy.loginfo('preprocessing_node started')
        self.lane = rospy.get_namespace().rstrip('/').split('/')[-1]
        self.camera_id_1 = rospy.get_param("~camera_id_1")
        self.camera_id_2 = rospy.get_param("~camera_id_2")
        self.bridge = CvBridge()
        self.output = inference()
        self.my_session = new_session("u2netp")
        #subscribers
        self.act_image = message_filters.Subscriber("actuator/image_raw", Image)
        self.non_act_image = message_filters.Subscriber("non_actuator/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.act_image,self.non_act_image],1,0.009)
        ts.registerCallback(self.image_callback)


    def image_callback(self, img1,img2):
        img_array1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding="rgb8")
        img_array2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding="bgr8")
        # size1, contour1 = self.run(img_array1)
        self.run(img_array1)
        cv2.imwrite("/home/agrograde/agrograde_ws/src/multilane_sorter/assets/2.jpg",img_array1)
        # print(size1)

    def run(self, img):
        output = remove(img,self.my_session)
        img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours= sorted(contours, key = cv2.contourArea, reverse = True)
        largest_contour = sorted_contours[0]
        #largest_contour = max(contours, key = cv2.contourArea)
        # ellipse = cv2.fitEllipse(largest_contour)
        contour_image = cv2.drawContours(img, [largest_contour], -1, (255,0,0), -1)
        # major_axis, minor_axis = ellipse[1]
        # sz = min(major_axis,minor_axis)
        # return sz, contour_image
        cv2.imwrite("/home/agrograde/agrograde_ws/src/multilane_sorter/assets/1.jpg",output)

if __name__ == '__main__':
    node = PreProcessing()    
    rospy.spin() 