import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelStates
import math
import cmath
import tf
from tf2_msgs.msg import TFMessage


class GazeboNavibotLidarNnEnv(gazebo_env.GazeboEnv):
    store=[0]
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboNavibotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/navibot/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.getmodelstate = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.reward_range = (-np.inf, np.inf)

        self.distance = 0
        self.angle = 0
        self._seed()

    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return np.array(data.ranges)/10,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # max_ang_speed = 0.3
        # ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        # vel_cmd.linear.x = 0.2
        # vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)
        if action == 1:
             vel_cmd.linear.x =0.5
        elif action == 2:
            vel_cmd.angular.z =0.2
            #vel_cmd.linear.x = 0.1
        elif action == 3:
            vel_cmd.angular.z =-0.2
            #vel_cmd.linear.x = 0.1
        else:
            vel_cmd.linear.x = -0.1

        self.vel_pub.publish(vel_cmd)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        robot_tf = None
        # while robot_tf is None:
        #     try:
        #         robot_tf = rospy.wait_for_message("/tf", TFMessage, timeout=5)
        #     except:
        #         pass
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)
# added code
#         targetPosition = self.getmodelstate
        # mobilePosition = self.getmodelstate####### ModelState?
        targetPositionRe =self.getmodelstate
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            # targetPosition = self.getmodelstate('Target', 'world')
            # mobilePosition = self.getmodelstate('mobile_base', 'world')
            targetPositionRe=self.getmodelstate("Target","chassis")
        except(rospy.ServiceException)as e:
            print('get the target position failed !')

        # x= targetPosition.pose.position.x-mobilePosition.pose.position.x
        # y= targetPosition.pose.position.y-mobilePosition.pose.position.y
        x=targetPositionRe.pose.position.x
        y=targetPositionRe.pose.position.y
        # distance,angle= cmath.polar(complex(x,y))
        # state = list(state)
        # state = state.extend([distance,angle])
        distance,theta=cmath.polar(complex(x,y))  #theta (-pi, pi)
        theta=np.arcsin(y/distance)
        if x < 0 and y < 0:
            # angle = -np.pi - np.arcsin(y / distance)
            angle =-np.pi-theta
        elif x < 0 and y > 0:
            angle= np.pi-theta
            # angle = np.pi - np.arcsin(y / distance)
        else:
            # angle = np.arcsin(y / distance)
            angle=theta
        # quaternion=(robot_tf.transforms[0].transform.rotation.x,
        #             robot_tf.transforms[0].transform.rotation.y,
        #             robot_tf.transforms[0].transform.rotation.z,
        #             robot_tf.transforms[0].transform.rotation.w)
        # euler=np.array(tf.transformations.euler_from_quaternion(quaternion))


        self.angle = np.round(abs(angle)/np.pi,2)
        self.distance = np.round(distance/10,2)
        anglereward = -self.angle
        # if (abs(targetPosition.pose.position.x - mobilePosition.pose.position.x) and
        #     abs(targetPosition.pose.position.y - mobilePosition.pose.position.y)) <= 0.3:
        if(distance<=0.3):
            target = True
            # print("targetPosition.pose.position.y"+str(targetPosition.pose.position.y))
            # print("mobilePosition.pose.position.y"+str(mobilePosition.pose.position.y))
        else:
            target = False

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            #reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
            if target:
                reward = 200
                done = True
            else:
                # distance2 = math.hypot(mobilePosition.pose.position.x - targetPosition.pose.position.x,
                #                        mobilePosition.pose.position.y - targetPosition.pose.position.y)
                #distance1 = GazeboMylabTurtlebotLidarNnEnv.store[0]
                reward= -self.distance
                #GazeboMylabTurtlebotLidarNnEnv.store[0] = distance
        else:
            reward = -200
        #print("reward is :"+str(reward))
        #print("distance2:"+str(distance2)+" distance1: "+str(distance1))

        # if abs(angle-euler[2])>np.pi:
        #     anglereward=-abs(angle-euler[2]-2*np.pi)/(np.pi)
        # else:
        #     anglereward=-abs(angle-euler[2])/(np.pi)
        # print("anglereward is: "+str(math.degrees(-abs(angle)))+"reward is :"+str(reward))
        reward=reward+anglereward
        # self.angle=np.round(angle-euler[2],2)/np.pi
        state=np.hstack([state,self.distance,self.angle])

        return state, reward, done, {}

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        # rospy.wait_for_service("/gazebo/get_model_state")
        # try:
        #     targetPosition = self.getmodelstate('Target', 'world')
        #     mobilePosition = self.getmodelstate('mobile_base', 'world')
        # except(rospy.ServiceException)as e:
        #     print('get the target position failed !')

        # x = targetPosition.pose.position.x - mobilePosition.pose.position.x
        # y = targetPosition.pose.position.y - mobilePosition.pose.position.y
        # distance, angle = cmath.polar(complex(x, y))
        # state=list(state)
        # state = state.extend([distance,angle])
        state=np.hstack([state,self.distance,self.angle])

        return np.asarray(state)
