#!/usr/bin/env python3

import glob
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import numpy as np
import skimage
from skimage import io
import os

class ImageLoad(object):
    def __init__(self):
        rospy.init_node('image_publisher')
        self.pub = rospy.Publisher('image_raw',Image,queue_size=1)
        self.rate =rospy.Rate(1)
        self.bridge = CvBridge()
        #self.com = rospy.Publisher('compressed_image_raw',CompressedImage,queue_size=1)
        #self.gui_msg = CompressedImage()

    def image_loop(self):  
        #while not rospy.is_shutdown(): 
            # self.img = io.imread('/home/agrograde/sayooj_ws/src/multilane_sorter/assets/sample/Double/3.jpg')
            # self.img =cv2.imread('/home/agrograde/sayooj_ws/src/multilane_sorter/assets/sample/Sprouting/528.jpg')
            # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) 
            # rospy.loginfo(type(self.img))
            # self.img =self.bridge.cv2_to_imgmsg(self.img,encoding='rgb8')
            # self.pub.publish(self.img)
        imgs = glob.glob("/home/agrograde/potato_ws/src/multilane_sorter/assets/images/lane_1/camera_11/*.jpg")
        while not rospy.is_shutdown():
        
            try:
                for img in imgs:
                    self.n= io.imread(img)
                    self.n= self.bridge.cv2_to_imgmsg(self.n,encoding='rgb8')
                    self.pub.publish(self.n)
                    # rospy.sleep(0.35)  
                    rospy.sleep(5)  
                    #rospy.loginfo(self.n)
                    #cv2.imshow(self.n) 
                    #self.guiimg = cv2.resize(self.n,(0,0),fx=0.2,fy=0.2)
                    #self.gui_msg.format = "jpeg"
                    #self.gui_msg.data = np.array(cv2.imencode('.jpeg', self.guiimg)[1]).tostring()
                    #self.com.publish(self.gui_msg) 
            except rospy.ROSInterruptException:
                rospy.logerr("ROS Interrupt Exception! Just ignore the exception!")
  
                      

if __name__ == '__main__':
    node = ImageLoad()
    node.image_loop() 
    rospy.spin()       