#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int8
from multilane_sorter.utils import signaling
from multilane_sorter.msg import Signal
import time
CGRN    = '\033[92m'
CRED    = '\033[91m'


class SignalingNode(object):
    def __init__(self):
        rospy.init_node('listener', anonymous=False)
        # rospy.loginfo("signaling node started")
        self.arduino_num = rospy.get_param('~arduino_num')
        rospy.Subscriber("decision_assign", Signal, self.callback)
        self.relay = signaling.RelaySignal(self.arduino_num,True )
        self.act = -1
        self.non_act = -1

    def callback(self,data):
        signal_time = time.time()
        # rospy.loginfo("data is coming")
        signal = [data.ac_a,data.ac_b,data.ac_c]
        # signal = [0,1,0]
        rospy.loginfo(signal)
        
        # signal = [1,0,0] 
        # rospy.loginfo(data.header.stamp)
        # print("current time is", rospy.Time.now())
        # rospy.loginfo("array which is sent to mcb is..")
        # rospy.loginfo(signal)
        self.relay.relay(signal)

        duration =rospy.Time.now() - data.header.stamp
        rospy.loginfo("duration is ::::: {} seconds".format(duration.to_sec()))
        rospy.loginfo("duration is ::::: {} seconds {} nseconds".format(duration.secs,duration.nsecs))
        rospy.loginfo(f'time taken for signal sending = {time.time()-signal_time} secs')
        # rospy.loginfo(duration.to_sec())
        
                           
if __name__ == '__main__':
    SignalingNode()
    rospy.spin()                      
   