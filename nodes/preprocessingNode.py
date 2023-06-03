#!/usr/bin/env python3

"""
Author: Vasan Naddoni and Vinayak Kunder
copyright @ Occipital Technologies Pvt Ltd
"""
from email.mime import image
import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from agrograde_msgs.msg import Preprocess 
import time
import message_filters


class PreProcessing(object):
    def __init__(self):
        rospy.init_node('preprocessing_node')
        rospy.loginfo('preprocessing_node started')
        self.lane = rospy.get_namespace().rstrip('/').split('/')[-1]
        self.camera_id = rospy.get_param("~camera_id")
        self.bridge = CvBridge()
        self.msg_packet = Preprocess()
        self.msg_saver = Preprocess()
        self.gui_msg = CompressedImage()

        self.kernel = np.ones((5,5),np.uint8)
        self.seg_x = 224
        self.seg_y = 168
        self.mask_threshold = rospy.get_param("~mask_threshold", 22)
        # Subscribers and Publishers
        try :
            non_actuator = rospy.Subscriber("~image_raw", Image, self.merge_images)
            # actuator = rospy.Subscriber("~actuator/image_raw", Image)
            # message_filters.ApproximateTimeSynchronizer([actuator, non_actuator],1,0.020).registerCallback(self.merge_images)
        except Exception as es:
            rospy.logerr(es) 
        
        self.post_processing_pub = rospy.Publisher('~post_processing', CompressedImage, queue_size=1)
        self.actual_images_pub =  rospy.Publisher("~actual_images",Preprocess,queue_size=10)
        self.cropped_images =  rospy.Publisher("~cropped_images",Preprocess,queue_size=10) #not publishing
        
        # Visualization
        self.viz_cropped_image_pub = rospy.Publisher('~viz/cropped_images', Image, queue_size=1)
        

    def getimgMask(self, img_inp):
        """
            This function generates mask of the input image using cv
            input: color image (BGR)
            output: binary mask
        """
        # generates mask of object with noise reduction
        img = cv2.cvtColor(img_inp, cv2.COLOR_BGR2GRAY)
        self.mask_threshold = rospy.get_param("~mask_threshold", 22)
        _,thresh = cv2.threshold(img, self.mask_threshold, 255, cv2.THRESH_BINARY)  # the min value has been set w.r.t. background enviornment
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.erode(mask, (5,5))
        mask = cv2.dilate(mask, (15,15))
        return mask


    def findObjects(self,cnts, segmented_img):
        dims_lst, cropped_imgs, centers = [], [], []
        # for c in cnts:
        c = cnts[0]
        x,y,w,h = cv2.boundingRect(c)
        center, dims, _ = cv2.minAreaRect(c)
        cropped = segmented_img #[y:y+h, x:x+w]
        dims_lst.append(int(max(dims)))
        # print("cropped_sum:", sum(cv2.mean(cropped)))
        cropped_imgs = cv2.resize(cropped,(self.seg_x,self.seg_y))
        # if len(cnts)==1:
        #     inp_image_cr = np.hstack((cropped_imgs[0], cropped_imgs[0]))
        # else:
        #     inp_image_cr = np.hstack((cropped_imgs[0], cropped_imgs[1]))
        return cropped_imgs, [max(dims_lst)]

    def get_cropped(self,img_mask,image):
        """
            This function finds object contours and crops the region
            input: binary mask , original image
            output: cropped image , maximum size of object in pixel

            note: here img_mask is 224x224 and image is 2560x720
        """
        #resize the mask back to original shape for full resolution segmentation
        mask=cv2.resize(img_mask,(image.shape[1],image.shape[0]))
        segmented_img = cv2.bitwise_and(image, image, mask=mask)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(contours)>1:
            return self.findObjects(cnts[:2], segmented_img)
        elif len(contours)==1:
            return self.findObjects(cnts, segmented_img)
        else:
            return None, None

    def preProcessing(self,image,msg):
        start = time.time()
        actual_image = image
        mask = self.getimgMask(cv2.resize(actual_image,(self.seg_x,self.seg_y)))

        result,length_data = self.get_cropped(mask,actual_image)
        if result is not None :
            self.msg_packet.length_data.data = length_data 
            self.msg_packet.lane = self.lane
            self.msg_packet.header.stamp = msg.header.stamp 
            #above 3 lines not required
            self.msg_packet.image = self.bridge.cv2_to_imgmsg(result,"bgr8")
            # self.cropped_images.publish(self.msg_packet)
            self.viz_cropped_image_pub.publish(self.msg_packet.image)
            result = cv2.copyMakeBorder(result,0,0,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))
            self.gui_msg.format = "jpeg"
            self.gui_msg.data = np.array(cv2.imencode('.jpeg', result)[1]).tobytes()
            self.post_processing_pub.publish(self.gui_msg)

            self.msg_saver.lane = self.lane
            self.msg_saver.full_image = self.bridge.cv2_to_imgmsg(actual_image,"bgr8")
            self.actual_images_pub.publish(self.msg_saver)
            print(f"time taken by preprocessing {self.lane} is {(time.time()- start)*1000} ")

    
    def merge_images(self,msg):
        image_array_actuator = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        path = "/home/agrograde/agrograde_ws/src/multilane_sorter/assets/images/{0}/{1}/".format(self.lane, self.camera_id)
        cv2.imwrite(path+str(msg.header.seq)+".jpg",image_array_actuator)
        # image_array_non_actuator = self.bridge.imgmsg_to_cv2(non_actuator,"bgr8")
        # value_ac = sum(cv2.mean(image_array_actuator))
        # value_nac = sum(cv2.mean(image_array_non_actuator))
        # if value_ac > 200:
        #     frame = np.hstack((np.zeros_like(image_array_actuator),image_array_non_actuator))
        #     # os.system(f"rosnode kill /{self.lane}/camera_actuator_node")
        # if value_nac > 200:
        #     frame = np.hstack((image_array_actuator,np.zeros_like(image_array_actuator)))
        #     # os.system(f"rosnode kill /{self.lane}/camera_non_actuator_node")

        # if value_nac < 200 and value_ac < 200:
        #     frame = np.hstack((image_array_actuator,image_array_non_actuator))
        self.preProcessing(image_array_actuator,msg)


    def image_acq(self,msg,lane):
        print(f"acquired from \033[92m {lane}\033[0m")


if __name__ == '__main__':
    node = PreProcessing()
    rospy.spin()
