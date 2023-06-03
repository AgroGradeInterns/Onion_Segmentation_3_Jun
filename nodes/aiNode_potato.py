#!/usr/bin/env python3


import os
import glob
import message_filters
import numpy as np
import pandas as pd
import rospy
import skimage
from skimage import io
import imageio

import torch, optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from torchvision.models import mobilenet_v2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Int8
import sys

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from torchmetrics import Precision, Recall, F1Score, Accuracy
from tqdm.auto import tqdm
from multilane_sorter.msg import inference
# from std_msgs.msg import Time
import time
import csv
from skimage.measure import label, regionprops, find_contours
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

        # self.time_pub = rospy.Publisher('image_receiving_time', Time, queue_size=1)
        self.onion = Net('/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/archive/hik_o_081222.pth')
        self.black_smut = Net('/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/hik_bs_071222.pth')
        self.peel = Net('/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/hik_p_161222.pth')
        self.multiplier = rospy.get_param('/sortobot/multiplier/'+self.lane)
        self.current_season = rospy.get_param('/sortobot/models/in_use')
        # self.model = Network("/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/hik_clf_161222.pth")

        self.model = Network("/home/agrograde/agrograde_ws/src/multilane_sorter/assets/hik_models/potato_clf_291222.pth")
        # self.time = Time()
    # def image_callback1(self,img1):
    #     
    #     self.image1 = img1
    # def image_callback2(self,img2):
    #     self.image2 = img2
    #     self.image_callback()

    def binarize(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(img, 14, 255, cv2.THRESH_BINARY)
        return alpha
    def floodfill(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        im_floodfill = img.copy()
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = img | im_floodfill_inv
        return im_out
    def spec_ff(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        im_floodfill = img.copy()
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        return im_floodfill
    def erode(self, img):
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 2)
        return erosion
    def contour(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((img.shape[0], img.shape[1]))
        sorted_contours= sorted(contours, key = cv2.contourArea, reverse = True)
        largest_item = sorted_contours[0]
        cv2.drawContours(mask, [largest_item], -1, (255,0,0), -1)
        mask = mask.astype(np.uint8)
        return mask
    def crop(self, img, mask):
        extracted = cv2.bitwise_and(img, img, mask = mask)
        return extracted
    def run(self, img, flag = None):
        binary = self.binarize(img)
        if flag == None:
            flood = self.floodfill(binary)
        else:
            flood = self.spec_ff(binary)
        erode = self.erode(flood)
        contour = self.contour(erode)
        crop = self.crop(img, contour)
        mask = cv2.resize(contour, (256, 256))
        return crop, mask

    def for_gui(self,c1,c2,img1,img2):
        bulb1 = Image()
        bulb_cv1         = np.zeros_like(img1)
        bulb_cv1[:,:,0]  = bulb_cv1[:,:,1] = bulb_cv1[:,:,2] = cv2.resize(c1.astype(np.uint8),(img1.shape[1],img1.shape[0]))
        bulb1 = self.bridge.cv2_to_imgmsg(bulb_cv1,"passthrough")
        bulb2 = Image()
        bulb_cv2         = np.zeros_like(img2)
        bulb_cv2[:,:,0]  = bulb_cv2[:,:,1] = bulb_cv2[:,:,2] = cv2.resize(c2.astype(np.uint8),(img2.shape[1],img2.shape[0]))
        bulb2 = self.bridge.cv2_to_imgmsg(bulb_cv2,"passthrough")
        self.mask_pub1.publish(bulb1)
        self.mask_pub2.publish(bulb2)

        
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
        path1 = "/home/agrograde/agrograde_ws/src/multilane_sorter/assets/images/{0}/{1}/".format(self.lane, self.camera_id_1)
        path2 = "/home/agrograde/agrograde_ws/src/multilane_sorter/assets/images/{0}/{1}/".format(self.lane, self.camera_id_2)
        io.imsave(path1+str(img1.header.seq)+"_"+str(self.output.prediction)+".jpg",img_array1)
        io.imsave(path2+str(img2.header.seq)+"_"+str(self.output.prediction)+".jpg",img_array2)
        # cv2.namedWindow("actuator",cv2.WINDOW_NORMAL)
        # cv2.imshow("actuator_img",img_array1)
        # cv2.waitKey(0)
        # rospy.loginfo(y)
    def message(self,array_1):
        try:
            array = [round(item, 2) for item in array_1]
        except:
            array = array_1
        try:
            array[4] = self.multiplier*array[4]
            self.output.prediction = array[0]
            self.output.onion = array[1]
            self.output.blacksmut = array[2]
            self.output.peel = array[3]
            self.output.size = array[4]
            self.ai_pub.publish(self.output)
            rospy.loginfo(self.output)
        except:
            pass
        # rospy.loginfo("the publishing data from preprocessing node is...")
    def decision(self, img_array1, img_array2):
        # rospy.loginfo(self.model)
        try:
            start = time.time()
            i1, c1 = self.run(img_array1)
            # path = "/home/agrograde/Desktop/"

            if self.camera_id_2 != "camera_22":
                i2, c2 = self.run(img_array2)
            else:
                i2, c2 = self.run(img_array2,flag =1)
                # cv2.imwrite(path+"contour.jpg",c)     
            y1 = self.model.predict(i1, i2)
            cl =time.time()
            # rospy.loginfo("time taken for classification is {} seconds".format(cl-start))  
            cond_time=time.time()
            if y1[0] == 1:
                #rejection
                # rospy.loginfo("REJECTED BY THE CLASSIFIER")
                array = y1
            #     # rospy.loginfo(array)
            elif y1[0] == 0:

                array = y1 #self.make_pred(img_array1,img_array2)
            else:
                pass
                # rospy.loginfo('no image caught') 
            rospy.loginfo(array)
            msg_time = time.time()
            # rospy.loginfo(f"time taken for conditional statement = {msg_time-cond_time}")    
            self.message(array)
            # self.for_gui(c1,c2,img_array1,img_array2)
            # rospy.loginfo(f"time for message {time.time()-msg_time}")
        except:
            pass
    def mask_to_border(self, mask):
        h, w = mask.shape
        border = np.zeros((h, w))
        contours = find_contours(mask, 128)
        for contour in contours:
            for c in contour:
                x = int(c[0])
                y = int(c[1])
                border[x][y] = 255
        return border

    # def calc_onion_sz(self, c1, c2):
    #     print("11111111111111111111111111111")
    #     sm1 = self.new_process(c1)
    #     print("2222222222222222222222222222222222")
    #     sm2 = self.new_process(c2)
    #     print("33333333333333333333333333333333333")
    #     size = (sm1 + sm2) / 2
    #     return size
    
    def get_big(self, bboxes):
        max_a = 0
        for bbox in bboxes:
            x = bbox[2] - bbox[0]
            y = bbox[3] - bbox[1]
            a = x * y
            if a > max_a:
                max_a = a
                final_bbox = bbox
        return final_bbox

    def mask_to_bbox(self, mask):
        bboxes = []
        mask = self.mask_to_border(mask)
        lbl = label(mask)
        props = regionprops(lbl)
        lns = []
        for prop in props:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]
            x2 = prop.bbox[3]
            y2 = prop.bbox[2]
            l = prop['MajorAxisLength']
            lns.append(l)
            bboxes.append([x1, y1, x2, y2])
        return bboxes, lns

    # def new_process(self, mask):
    #     bboxes = self.mask_to_bbox(mask)
    #     bbox = self.get_big(bboxes)
    #     d1 = bbox[2] - bbox[0]
    #     d2 = bbox[3] - bbox[1]
    #     D = np.sqrt(d1**2 + d2**2)
    #     return D

    def process_mask(self, mask):
        _, threshed = cv2.threshold(mask, 1, 2, cv2.THRESH_BINARY)
        threshed /= 2.0
        threshed *= 255
        threshed = np.asarray(threshed, dtype = np.uint8)
        bboxes, lns = self.mask_to_bbox(threshed)
        # bbox = self.get_big(bboxes)
        # d1 = bbox[2] - bbox[0]
        # d2 = bbox[3] - bbox[1]
        # D = np.sqrt(d1**2 + d2**2)
        l = max(lns)
        return l
    def stats(self,o1, o2, p1, p2, b1, b2):
        # try:
        y1 = o1 + p1 + b1
        y2 = o2 + p2 + b2
        skin1 = np.count_nonzero(np.round(o1))
        skin2 = np.count_nonzero(np.round(o2))
        skin = (skin1 + skin2) / 2
        peel1 = np.count_nonzero(np.round(p1))
        peel2 = np.count_nonzero(np.round(p2))
        peel = (peel1 + peel2) / 2
        black_smut1 = np.count_nonzero(np.round(b1))
        black_smut2 = np.count_nonzero(np.round(b2))
        black_smut = (black_smut1 + black_smut2) / 2
        area = skin + peel + black_smut
        sm1 = self.process_mask(y1)
        sm2 = self.process_mask(y2)
        size = (sm1 + sm2)/2
        # d =  2 * np.sqrt(area / np.pi)

        pct_skin = 100 * (skin / area)
        pct_black_smut = 100 * (black_smut / area)
        pct_peel = 100 * (peel / area)
        return np.array([0, pct_skin, pct_black_smut, pct_peel, size])
    #    Taking Max
    # def stats(self,o1, o2, p1, p2, b1, b2):
    #     # try:
    #     y1 = o1 + p1 + b1
    #     # print(f"****************{y1}")
    #     y2 = o2 + p2 + b2
    #     # print(f"****************{y2}")
    #     # print(f"y1: {np.sum(y1)}, y2: {np.sum(y2)}")
    #     if np.sum(y1) > np.sum(y2):
    #         l = self.process_mask(y1)
    #         skin = np.count_nonzero(np.round(o1))
    #     else:
    #         l = self.process_mask(y2)
    #         skin = np.count_nonzero(np.round(o2))
    #     # print(f"****************{l}")
    #     # print(f"****************{skin}")
    #     peel1 = np.count_nonzero(np.round(p1))
    #     peel2 = np.count_nonzero(np.round(p2))
    #     peel = (peel1 + peel2) / 2
    #     # print(f"****************{peel}")
    #     black_smut1 = np.count_nonzero(np.round(b1))
    #     black_smut2 = np.count_nonzero(np.round(b2))
    #     black_smut = (black_smut1 + black_smut2) / 2
    #     # print(f"****************{black_smut}")
    #     area = skin + peel + black_smut
    #     # print(f"****************{area}")
    #     pct_skin = 100 * (skin / area)
    #     # print(f"****************{skin}")
    #     pct_black_smut = 100 * (black_smut / area)
    #     # print(f"****************{pct_black_smut}")
    #     pct_peel = 100 * (peel / area)
    #     # print(f"****************{pct_peel}")
    #     # print(f"onion: {pct_skin}, black_smut: {pct_black_smut}, peel: {pct_peel}")
        
    #     # data = {
    #     #         "size" : str(l*0.345),
    #     #         "y1" : str(np.sum(y1)),
    #     #         "y2" : str(np.sum(y2))
    #     #     }
    #     # filepath = "/home/agrograde/"+self.lane+".csv"
    #     # fileexists = os.path.isfile(filepath)
    #     # with open(filepath, mode='a') as csv_file:
    #     #     fieldnames = data.keys()
    #     #     self.writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     #     if not fileexists:
    #     #         self.writer.writeheader()
    #     #     if self.writer is not None:
    #     #         self.writer.writerow(data)

    #     return np.array([0, pct_skin, pct_black_smut, pct_peel, l])
        # except:
        #     pass
    # def stats(self, o1, o2, p1, p2, b1, b2):
    #     try:
    #         y1 = o1 + p1 + b1
    #         y2 = o2 + p2 + b2
    #         skin1 = np.count_nonzero(np.round(o1))
    #         skin2 = np.count_nonzero(np.round(o2))
    #         skin = (skin1 + skin2) / 2
    #         # print("$$$$$$$$$$$$$$$$$$$$$$$$$")
    #         # print(o1.shape)
    #         # print(o2.shape)
    #         # print(p1.shape)
    #         # print(p2.shape)
    #         # print(b1.shape)
    #         # print(b2.shape)
    #         # sz = self.calc_onion_sz(o1, o2)
    #         # print(f"{sz}")
    #         # onion = (np.count_nonzero(o1) + np.count_nonzero(o2)) / 2
    #         # print(f"{onion}")
    #         peel1 = np.count_nonzero(np.round(p1))
    #         peel2 = np.count_nonzero(np.round(p2))
    #         peel = (peel1 + peel2) / 2
    #         # print(f"{peel}")
    #         black_smut1 = np.count_nonzero(np.round(b1))
    #         black_smut2 = np.count_nonzero(np.round(b2))
    #         black_smut = (black_smut1 + black_smut2) / 2
    #         # print(f"{black_smut}")
    #         area = skin + peel + black_smut
    #         sm1 = self.process_mask(y1)
    #         sm2 = self.process_mask(y2)
    #         size = (sm1 + sm2)/2
    #         # c = 
    #         d =  2 * np.sqrt(area / np.pi)
    #         pct_black_smut = 100 * ((onion - (onion - black_smut)) / onion)
    #         # pct_peel = 100 * ((onion - (onion - peel)) / onion)
    #         # pct_onion = 100 - (pct_black_smut + pct_peel)
    #         # print(f"22222222222222{pct_black_smut}, {pct_peel}, {pct_onion}")
    #         return np.array([0, pct_onion, pct_black_smut, pct_peel, sz])
        # except:
        #     pass
#     print(f'Skin: {pct_skin:.3f}%, Peel: {pct_peel:.3f}%, Black Smut: {pct_black_smut:.3f}%')

# stats(o, p, b)

    def make_pred(self, img1, img2):
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # o1, o2 = self.onion.predict(img1, img2)
        o1, o2 = self.onion.predict(img1, img2)
        # np.save("/home/agrograde/Desktop/img_1__make_pred.npy", img1)
        # np.save("/home/agrograde/Desktop/img_2__make_pred.npy", img2)
        o1, o2 = o1.cpu().numpy(), o2.cpu().numpy()
        # _, o2 = self.run(img2)
        b1, b2 = self.black_smut.predict(img1, img2)
        p1, p2 = self.peel.predict(img1, img2)
        b1, b2 = b1.cpu().numpy(), b2.cpu().numpy()
        p1, p2 = p1.cpu().numpy(), p2.cpu().numpy()
        # cv2.namedWindow("actuator",cv2.WINDOW_NORMAL)
        # cv2.imshow("actuator_img",np.squeeze(p1))
        # cv2.waitKey(0)

        s = self.stats(o1, o2, p1, p2, b1, b2)
        return s
class Net(nn.Module):
    def __init__(self, path):
        super(Net, self).__init__()
        self.model = smp.UnetPlusPlus(
        encoder_name = 'resnet18',
        encoder_weights = None,
        in_channels = 3,
        classes = 1,
        activation = 'sigmoid')
        if torch.cuda.is_available():
            self.device = "cuda:0"
            torch.cuda.empty_cache()
        else:
            self.device = "cpu"
        self = self.to(self.device)
        self.load_state_dict(torch.load(path))
        self.tfms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean = 0.0, std = 1.0),
            ToTensorV2()
        ])
    def predict(self, x1, x2):
        self.eval()
        # np.save("/home/agrograde/Desktop/img_1__seg.npy", x1)
        # np.save("/home/agrograde/Desktop/img_2__seg.npy", x2)
        x1 = self.tfms(image = x1)['image'].to(self.device)
        x1 = torch.unsqueeze(x1, axis = 0)
        x2 = self.tfms(image = x2)['image'].to(self.device)
        x2 = torch.unsqueeze(x2, axis = 0)
        with torch.no_grad():
            y1 = self(x1)
            y2 = self(x2)
        return torch.squeeze(y1), torch.squeeze(y2)
        
    def forward(self, x):
        y = self.model(x)
        return y    
                 

        
class Network(nn.Module):
    def __init__(self,path):
        super(Network, self).__init__()
        self.model = mobilenet_v2(weights = None)
        self.model.classifier = nn.Sequential(nn.Linear(1280, 512),
                                              nn.Linear(512, 256),
                                              nn.Linear(256, 64),
                                              nn.Linear(64, 1),
                                              nn.Sigmoid())
        if torch.cuda.is_available():
            self.device = "cuda:0"
            torch.cuda.empty_cache()
        else:
            self.device = "cpu"
        self = self.to(self.device)
        self.load_state_dict(torch.load(path))
        self.tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean = 0.0, std = 1.0),
            ToTensorV2()
        ])
    def decide(self, y1, y2):
        print (f"y1: {y1}, y2: {y2}")
        if y1 >= 0.98 or y2 >= 0.98:
            y = np.array([1, 0, 0, 0, 0])
        else:
            y = np.array([0, 0, 0, 0, 0])
        return y
    # def decide(self, Y):
    #     if np.round(Y) == 1:
    #         y = np.array([1, 0, 0, 0, 0])
    #     else:
    #         y = np.array([0, 0, 0, 0, 0])
    #     return y
    def predict(self, x1, x2):
        # x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2RGB) #
        # x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB) #
        # np.save("/home/agrograde/Desktop/img_1__clf.npy", x1)
        # np.save("/home/agrograde/Desktop/img_2__clf.npy", x2)
        x1 = self.tfms(image = x1)['image'].to(self.device)
        x1 = torch.unsqueeze(x1, axis = 0)
        x2 = self.tfms(image = x2)['image'].to(self.device)
        x2 = torch.unsqueeze(x2, axis = 0)
        self.eval()
        with torch.no_grad():
            y1 = self(x1).cpu().numpy()
            y2 = self(x2).cpu().numpy()
        y = self.decide(y1, y2)
        return y
    # def predict(self, x1, x2):
    #     # x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2RGB) 
    #     # x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB) 
    #     x1 = self.tfms(image = x1)['image'].to(self.device)
    #     x1 = torch.unsqueeze(x1, axis = 0)
    #     x2 = self.tfms(image = x2)['image'].to(self.device)
    #     x2 = torch.unsqueeze(x2, axis = 0)
    #     self.eval()
    #     with torch.no_grad():
    #         y1 = float(self(x1).cpu().numpy())
    #         y2 = float(self(x2).cpu().numpy())    
    #     y = (y1 + y2) / 2
    #     y = self.decide(y)
    #     return y
    def forward(self, x):
        y = self.model(x)
        return y

class get_crop():
    def __init__(self):
        pass
     

if __name__ == '__main__':
    node = PreProcessing()    
    rospy.spin() 