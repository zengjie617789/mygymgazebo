import rospy
from gazebo_msgs.msg import ContactsState
import subprocess
import numpy as np
# pathLaunchfile="/home/control511/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/" \
#                "src/turtlebot_simulator/turtlebot_gazebo/launch/myworld.launch"
# subprocess.Popen(["roslaunch", pathLaunchfile])

# def callback(data):
#     state=data.states
#     print("state is ")
#     print(state[i].collision2_name.split("::")[0] !="cylinder" for i in range(len(state)))
#     # print(state[1])
#     # if len(state)>0:
#         # print([state[i].collision2_name.split('::')[0]
#         #                      for i in range(len(state))])
#
#
#
# try:
#     # base_link_dumper=rospy.wait_for_message("/turtlebot/base_link_bumper",ContactsState,timeout=5)
#     rospy.init_node("listener")
#     rospy.Subscriber("/turtlebot/base_link_bumper",ContactsState,callback)
#     rospy.spin()
#     # print(base_link_dumper)
# except:
#     raise rospy.exceptions.ROSException
# a=np.array([1,2])
# b=np.array([3,4])
# c=0
# d=3
#
# data=np.hstack((a,b,c,d))
# data2=np.hstack((a,[c,d],b,1))
# print(data2)
# print(data2[:2])
# print(data2[2])
# print(data2[3])
# print(data2[4:-1])
# if 1 == True:
#     print("true")
# else:
#     print("false")
a=0
for i in range(3):
   a+=1
   if a==2:
       continue
   print(a)