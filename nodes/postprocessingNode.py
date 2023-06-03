#!/usr/bin/env python3


import os
import rospy
import glob
import message_filters
from sensor_msgs.msg import Image,CompressedImage
import numpy as np
import cv2
from cv_bridge import CvBridge

class PostProcessing(object):
    def __init__(self):
        rospy.init_node("postprocessing_node")
        rospy.loginfo("postprocessing node started")
        self.bridge =CvBridge()

        self.act_image = message_filters.Subscriber("actuator/image_raw", Image)
        self.non_act_image = message_filters.Subscriber("non_actuator/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.act_image,self.non_act_image],1,0.009)
        ts.registerCallback(self.image_callback)
  

        self.act_mask = message_filters.Subscriber("preprocessing_act",Image)
        self.non_act_mask = message_filters.Subscriber("preprocessing_non_act",Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.act_mask,self.non_act_mask],1,0.0009)
        ts.registerCallback(self.gui_fun)
      
        self.ppNode1 = rospy.Publisher('~postprocessing_act', CompressedImage, queue_size=1)
        self.ppNode2 = rospy.Publisher('~postprocessing_non_act', CompressedImage, queue_size=1)

    def image_callback(self,img1,img2):
        self.img1 = self.bridge.imgmsg_to_cv2(img1, "bgr8")
        self.img2 = self.bridge.imgmsg_to_cv2(img2, "bgr8")

    def gui_fun(self,msg1,msg2):
        gui_msg1 =CompressedImage()
        mask1 = self.bridge.imgmsg_to_cv2(msg1, "passthrough")
        guiimg1 = cv2.bitwise_and(self.img1, self.img1, mask=mask1[:,:,0])
        guiimg1 = cv2.resize(guiimg1,(0,0),fx=0.2,fy=0.2)
        # guiimg = bridge.cv2_to_imgmsg(guiimg,encoding="bgr8")
        gui_msg1.format = "jpeg"
        gui_msg1.data = np.array(cv2.imencode('.jpeg', guiimg1)[1]).tobytes()

        gui_msg2 = CompressedImage()
        mask2 = self.bridge.imgmsg_to_cv2(msg2, "passthrough")
        guiimg2 = cv2.bitwise_and(self.img2, self.img2, mask=mask2[:,:,0])
        guiimg2 = cv2.resize(guiimg2,(0,0),fx=0.2,fy=0.2)
        # guiimg = bridge.cv2_to_imgmsg(guiimg,encoding="bgr8")
        gui_msg2.format = "jpeg"
        gui_msg2.data = np.array(cv2.imencode('.jpeg', guiimg2)[1]).tobytes()    
        # guiimg = 
        
        self.ppNode1.publish(gui_msg1)
        self.ppNode2.publish(gui_msg2) 


if __name__ == "__main__":
    node = PostProcessing()
    rospy.spin()