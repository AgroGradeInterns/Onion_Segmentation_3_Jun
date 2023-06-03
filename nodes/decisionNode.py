#!/usr/bin/env python3

import rospy
import numpy as np 
import rospy
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from multilane_sorter.msg import Signal
from multilane_sorter.msg import inference
import time

CRED    = '\033[91m'
CGRN    = '\033[92m'
CBLNK   = '\33[5m'
CEND    = '\033[0m'
CREDBG  = '\33[41m'


class DecisionNode(object):

    def __init__(self):
        rospy.init_node('decision_node')
        # rospy.loginfo('decision_node started')
        self.lane = rospy.get_namespace().rstrip('/').split('/')[-1]
        self.dummy_array = [0,0,0]
        self.updateParams()
        self.output = Signal()
        rospy.set_param('/gui/decisionNodeUpdateFlag',0)
        rospy.set_param("/gui/resetsortoparam",0)

        # # Subscriber 
        rospy.Subscriber('ai_inference_channel', inference, self.decision)
        # # Publisher
        self.decision_publisher = rospy.Publisher("decision_assign",Signal,queue_size=1)
    def get_params(self): 
        if int(rospy.get_param('/gui/decisionNodeUpdateFlag')):
            self.updateParams() 
            rospy.set_param('/gui/decisionNodeUpdateFlag',0)
        if int(rospy.get_param("/gui/resetsortoparam")):
            self.reset_params()
            rospy.set_param("/gui/resetsortoparam",0)
    def updateParams(self):
        """
        This function reads and assigns all the necessary values from parameters.yaml and decision.yaml 
        """
        rospy.set_param('/gui/decisionNodeUpdateFlag',0)
        rospy.set_param("/gui/resetsortoparam",0)

        # self.multiplier = int(rospy.get_param('{}/multiplier'.format(self.lane)))
        self.current_season = rospy.get_param('/sortobot/models/in_use')
        self.current_quality = rospy.get_param('/sortobot/config/currentCriteria')
        self.sprout_tolerence = float(rospy.get_param('/decisionParams/defectSorting/tolerances/'+self.current_season+'/'+self.current_quality+'/defect/def_0'))
        self.blacksmut_tolerence = float(rospy.get_param('/decisionParams/defectSorting/tolerances/'+self.current_season+'/'+self.current_quality+'/defect/def_1'))
        self.rotten_tolerence = float(rospy.get_param('/decisionParams/defectSorting/tolerances/'+self.current_season+'/'+self.current_quality+'/defect/def_2'))
        # rospy.loginfo(self.peel_tolerence)
        self.size_min = int(rospy.get_param('/decisionParams/decision/size/A/min')) + int(rospy.get_param("/sortobot/size_offset"))
        self.size_max = int(rospy.get_param('/decisionParams/decision/size/A/max')) + int(rospy.get_param("/sortobot/size_offset"))
        self.range_min = 25
        self.range_max = 120
        rospy.loginfo(CREDBG+"parameters updated"+CEND)

    def reset_params(self):
        rospy.set_param('/decisionParams/defectSorting/tolerances/',rospy.get_param('/decisionParams/defectSorting/default/'))        
          
    def size_check(self,size):
        if size in range(self.size_min,self.size_max):
            array = [1,0,0]
            rospy.loginfo(CGRN+"BIN-A"+CEND)
        else:
            array = [0,1,0]
            rospy.loginfo(CGRN+"BIN-B"+CEND)
        return array    
    def size_range(self,size): 
        if size in range(25, 120):
            # rospy.loginfo("size is in range. given to check which bin it belongs to")
            array = self.size_check(size)
        else:
            # rospy.loginfo("size is out of range")
            array = [0,0,0]  
        return array              
    def decision(self,msg):
        #if parameters are updated from the GUI
        start = time.time()
        self.get_params()    
        if msg.sprout > self.sprout_tolerence:
            self.dummy_array = [0,0,0]
        elif msg.blacksmut > self.blacksmut_tolerence:
            self.dummy_array = [0,0,0]
             
        elif msg.rotten > self.rotten_tolerence:
            self.dummy_array = [0,0,0] 
        else:
            # rospy.loginfo("size is given to check in normal range")
            self.dummy_array = self.size_range(int(msg.size))
        self.output.header.stamp = msg.header.stamp

        if self.dummy_array == [0,0,0]:
            rospy.loginfo(CRED+"Rejected"+CEND)     
        self.update(self.dummy_array)
        # rospy.loginfo("time taken for decision node {} ".format(time.time()-start))

               
    def update(self,dummy_array):       
        self.output.ac_a = int(dummy_array[0])
        self.output.ac_b = int(dummy_array[1])
        self.output.ac_c = int(dummy_array[2])
        # rospy.loginfo("the output from decision node is...")
        # rospy.loginfo(self.output)
        self.decision_publisher.publish(self.output)
     
        


if __name__ == '__main__':
    # try:
    start_node = DecisionNode()
    # start_node.decision([0,1,2,3,56])
    rospy.spin()
    # except rospy.ROSInterruptException:
        # pass
