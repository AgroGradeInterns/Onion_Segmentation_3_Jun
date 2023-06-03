#!/usr/bin/env python3


import message_filters
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Int8
from multilane_sorter.msg import inference
# from std_msgs.msg import Time
import time
from skimage import io



import numpy as np
import cv2
import os
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,UpSampling2D,concatenate
import time

from keras.layers import *
from keras.models import *




CRED    = '\033[91m'
CGRN    = '\033[92m'
CBLNK   = '\33[5m'
CEND    = '\033[0m'
CREDBG  = '\33[41m'


class PreProcessing():
    def __init__(self):
        rospy.init_node('preprocessing_node')
        # rospy.loginfo('preprocessing_node started')
        self.lane = rospy.get_namespace().rstrip('/').split('/')[-1]
        self.camera_id_1 = rospy.get_param("~camera_id_1")
        self.camera_id_2 = rospy.get_param("~camera_id_2")
        self.bridge = CvBridge()
        self.output = inference()
        #subscribers
        self.act_image = message_filters.Subscriber("actuator/image_raw", Image)
        self.non_act_image = message_filters.Subscriber("non_actuator/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.act_image,self.non_act_image],1,0.009)
        ts.registerCallback(self.image_callback)
        # rospy.Subscriber("actuator/image_raw", Image, self.image_callback1)
        # rospy.Subscriber("non_actuator/image_raw", Image, self.image_callback2)

        #publishers
        self.ai_pub = rospy.Publisher('ai_inference_channel', inference, queue_size=1)
        self.mask_pub1 = rospy.Publisher('preprocessing_act',Image,queue_size=1)
        self.mask_pub2 = rospy.Publisher('preprocessing_non_act',Image,queue_size=1)

        # self.onion = Net('/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/archive/hik_o_081222.pth')
        # self.black_smut = Net('/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/hik_bs_071222.pth')
        # self.peel = Net('/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/hik_p_161222.pth')
        self.multiplier = rospy.get_param('/sortobot/multiplier/'+self.lane)
        self.current_season = rospy.get_param('/sortobot/models/in_use')
        #self.model = Segmentation_model(w_path="/home/agrograde/potato_ws/src/multilane_sorter/ai_models/M2_Onion_seg_25th_May_25May2023.h5")
        self.model = Segmentation_model(w_path="/home/agrograde/potato_ws/src/multilane_sorter/ai_models/M2_Onion_seg_25th_May_25May2023.h5")

        # self.model = Network("/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/potato_clf_291222.pth")
        # self.time = Time()
    # def image_callback1(self,img1):
    #     
    #     self.image1 = img1
    # def image_callback2(self,img2):
    #     self.image2 = img2
    #     self.image_callback()

      
    def image_callback(self,img1,img2):
        self.output.header.stamp = rospy.Time.now()
        # rospy.loginfo(self.output.header.stamp)
        # rospy.loginfo(img1.header.stamp)
        # duration = (rospy.Time.now() - img2.header.stamp)
        # rospy.loginfo("camera taking {} seconds".format(duration.to_sec()))
        rgb_time =time.time()
        img_array1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding="bgr8")
        img_array2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding="bgr8")
        # print(f"*******************************{type(img_array1)}")
        # print(f"*******************************{type(img_array2)}")
        # np.save("/home/agrograde/Desktop/img_1__desc.npy", img_array1)
        # np.save("/home/agrograde/Desktop/img_2__desc.npy", img_array2)
        # rospy.loginfo(f"time taken for rgb conversion {time.time()-rgb_time}")       
        self.decision(img_array1, img_array2)
        path1 = "/home/agrograde/potato_ws/src/multilane_sorter/assets/images/{0}/{1}/".format(self.lane, self.camera_id_1)
        path2 = "/home/agrograde/potato_ws/src/multilane_sorter/assets/images/{0}/{1}/".format(self.lane, self.camera_id_2)
        io.imsave(path1+str(img1.header.seq)+"_"+".jpg",img_array1)
        io.imsave(path2+str(img2.header.seq)+"_"+".jpg",img_array2)
        # cv2.namedWindow("actuator",cv2.WINDOW_NORMAL)
        # cv2.imshow("actuator_img",img_array1)
        # cv2.waitKey(0)
        # rospy.loginfo(y)
    def message(self,array_1):

        array = [round(item, 2) for item in array_1]
    
        array[3] = self.multiplier*array[3]
        self.output.sprout = array[0]
        self.output.blacksmut= array[1]
        self.output.rotten = array[2]
        self.output.size = array[3]
        self.ai_pub.publish(self.output)
        rospy.loginfo(self.output)
       
        # rospy.loginfo("the publishing data from preprocessing node is...")
    def decision(self, img_array1, img_array2):
        # rospy.loginfo(self.model)
        #array = [sprout,black_smut,rotten,size]
        array = self.model.get2_img(img_array1,img_array2)  
        self.message(array)
        

class Segmentation_model():
    def __init__(self,w_path):
        

        self.w_path = w_path
        self.masks =[]
        self.output_model()
      
    def SEModule(self,input, ratio, out_dim):
        # bs, c, h, w
        x = GlobalAveragePooling2D()(input)
        excitation = Dense(units=out_dim // ratio)(x)
        excitation = Activation('relu')(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1, 1, out_dim))(excitation)
        scale = multiply([input, excitation])
        return scale   
        
    def SEUnet(self,nClasses, input_height=224, input_width=224):
        inputs = Input(shape=(input_height, input_width, 3))
        conv1 = Conv2D(16,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)

        conv1 = Conv2D(16,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)

        # se
        conv1 = self.SEModule(conv1, 4, 16)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)

        conv2 = Conv2D(32,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)

        # se
        conv2 = self.SEModule(conv2, 8, 32)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)

        conv3 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)

        # se
        conv3 = self.SEModule(conv3, 8, 64)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)

        conv4 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)

        # se
        conv4 = self.SEModule(conv4, 16, 128)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)

        # se
        conv5 = self.SEModule(conv5, 16, 256)

        up6 = Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv5))
        up6 = BatchNormalization()(up6)

        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)

        conv6 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        # se
        conv6 = self.SEModule(conv6, 16, 128)

        up7 = Conv2D(64,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
        up7 = BatchNormalization()(up7)

        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)

        conv7 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        # se
        conv7 = self.SEModule(conv7, 8, 64)

        up8 = Conv2D(32,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
        up8 = BatchNormalization()(up8)

        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(32,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)

        conv8 = Conv2D(32,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        # se
        conv8 = self.SEModule(conv8, 4, 32)

        up9 = Conv2D(16,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
        up9 = BatchNormalization()(up9)

        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(16,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)

        conv9 = Conv2D(16,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)

        # se
        conv9 = self.SEModule(conv9, 2, 16)

        conv10 = Conv2D(nClasses, (3, 3), padding='same')(conv9)
        conv10 = BatchNormalization()(conv10)

        outputHeight = Model(inputs, conv10).output_shape[1]
        outputWidth = Model(inputs, conv10).output_shape[2]

        out = (Reshape((outputHeight * outputWidth, nClasses)))(conv10)
        out = Activation('softmax')(out)

        model = Model(inputs, out)
        model.outputHeight = outputHeight
        model.outputWidth = outputWidth

        return model
    
    def output_model(self):
    
        model1 = self.SEUnet(nClasses=4)
        x = model1.get_layer(index=-3).output
        out0 = Conv2D(2, (1, 1), activation='softmax',name='Sprout')(x)
        out1 = Conv2D(2, (1, 1), activation='softmax',name='Black_smut')(x)
        out2 = Conv2D(2, (1, 1), activation='softmax',name='Rotten')(x)#Conv2D(2, (1, 1), activation='softmax',name='rotten')(x)
        out3 = Conv2D(2, (1, 1), activation='softmax',name='Background')(x)

        self.model_new = Model(inputs = model1.input,outputs = [out0,out1,out2,out3])
        self.model_new.load_weights(self.w_path)
        
    def predict(self,image):
        
        result = self.model_new.predict(image)
        
        return result
    
    
    def getPercentArea(self, full_mask, region_mask):

        total_area = np.dot(full_mask.flatten(), np.ones_like(full_mask.flatten()))
        region_area = np.dot(region_mask.flatten(), np.ones_like(region_mask.flatten()))

        area_percentage = (region_area/total_area)*100

        return area_percentage
    

    def get2_img(self,img_path1,img_path2):

        defects = []
        s1,s2=0,0
        l1,s1=self.getPrediction_values(img_path1)
        l2,s2=self.getPrediction_values(img_path2)
        
        
        for i,j in zip(l1,l2):
            add = i+j
            print(type(add))
            average = add/2
            defects.append(average)

        size = max(s1,s2)
        print(defects)
        print(size)
        defects.append(size)
        return defects

        # return [1,1,1,1]
        # return f'final % are {avg_2_img} and size is {(max(s1,s2))}' #.format(avg_2_img,(max(s1,s2)))
    
    
    
    
        
    def getPrediction_values(self,img_path):
        h,w = 224,224
        

        im = cv2.resize(img_path,(h,w))

        I = im.reshape([1,h,w,3])
        start = time.time()
        
        
        preds = self.predict(I)
        
        
        sp = np.argmax(preds[0], axis=3)
        sp = sp.reshape([h,w])
       
        bs = np.argmax(preds[1], axis=3)
        bs = bs.reshape([h,w])
        
        ro = np.argmax(preds[2], axis=3)
        ro = ro.reshape([h,w])
        
        bg = np.argmax(preds[3], axis=3)
        bg = bg.reshape([h,w])
        

#         pl = np.argmax(preds[3], axis=3)
#         pl = pl.reshape([h,w])

        im = cv2.cvtColor(im , cv2.COLOR_BGR2RGB)

       
        all_masks = [sp,bs,ro,bg]  
      
        

        
        image_2d = cv2.convertScaleAbs(bg)
        image_rgb = np.stack((image_2d,) * 3, axis=-1)
        
        
        
       
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        try:
        # find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # fit ellipse
            ellipse = cv2.fitEllipse(contours[0])

            # calculate size
            size = min(ellipse[1]) * 0.08 #multiplier
    #         print("Object size: ye hai bro ", size)
        except:
            size=1

#         we were using binary 1 now we have changed to gray
        sprout_area = self.getPercentArea(gray, sp)#we were using binary 1 now we have changed to gray

        black_smut_area = self.getPercentArea(gray, bs)#we were using binary 1 now we have changed to gray

        rotten_area = self.getPercentArea(gray, ro)#we were using binary 1 now we have changed to gray
        
        background_area = self.getPercentArea(gray, bg) #we were using binary 1 now we have changed to gray

        total_area = rotten_area+black_smut_area+sprout_area
        # total_area = background_area

        r1,r2,r3=((sprout_area*100)/total_area),((black_smut_area*100)/total_area),((rotten_area*100)/total_area)
    
        final_percentage_features = [r1,r2,r3]
        
        return final_percentage_features,size
    
    def myfunct(self,path1,path2):
        for image in os.listdir(path1):
            img1 = cv2.imread(os.path.join(path1,image))

        for image in os.listdir(path2):
            img2 = cv2.imread(os.path.join(path2,image))    

        return self.get2_img(img1,img2)


     

if __name__ == '__main__':
    node = PreProcessing()    
    rospy.spin() 