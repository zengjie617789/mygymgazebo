import gym
import rospy
import roslaunch
import time
import numpy as np
import tensorflow as tf
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist ,Pose
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding
from gazebo_msgs.srv import GetModelState,SetModelState,SpawnModel
from gazebo_msgs.msg import ModelStates,ModelState,ContactsState
import math
import cmath
import tf
from tf2_msgs.msg import TFMessage


class GazeboMylabTurtlebotLidarNnEnv(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMylabTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.getmodelstate = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.setmodelstate = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self.spawn_goal=rospy.ServiceProxy('/gazebo/spawn_sdf_model',SpawnModel)
        self.reward_range = (-np.inf, np.inf)


        self._seed()

        self.goallist=np.array([[0,2,0],
                               [2,2,0],
                               [-2,1,0]])
        self.spawnGoal()

    def getAngelDistance(self,x,y):

        distance,theta=cmath.polar(complex(x,y))
        return  theta,distance

    def setRodomModelState(self,name):
        idx=np.random.randint(len(self.goallist)-1)
        goalPos=self.goallist[idx]

        state=ModelState()
        state.model_name=name
        state.pose.position.x=goalPos[0]
        state.pose.position.y=goalPos[1]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.setmodelstate(state)
        except rospy.ROSException as e:
            print("call service failed!")
    def spawnGoal(self):
        idx=np.random.randint(0,len(self.goallist)-1)
        goalPos=self.goallist[idx]
        goal_pose=Pose()
        goal_pose.position.x=goalPos[0]
        goal_pose.position.y=goalPos[1]

        PATH_GOAL = "/home/control511/gym-gazebo/gym_gazebo/envs/assets/models/cylinder/model.sdf"
        with open(PATH_GOAL,'r') as f:
            data=f.read()
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            self.spawn_goal("cylinder",data,"cyliderNameSpace",goal_pose,'world')
        except:
            raise rospy.exceptions.ROSException




    # def calculate_observation(self,data):
    #     min_range = 0.2
    #     done = False
    #
    #     for i, item in enumerate(data.ranges):
    #         if (min_range > data.ranges[i] > 0):
    #             done = True
    #     data=np.array(data.ranges)/np.sqrt(200)
    #     # data[np.where(np.isinf(data))[0]]=0
    #
    #     return data,done

    # def calculate_observation(self,data):
    #     done=False
    #     target=False
    #     data=np.array(data.ranges)/np.sqrt(200)
    #     try:
    #         base_link_bumper=rospy.wait_for_message('/turtlebot/base_link_bumper',ContactsState,timeout=5)
    #     except:
    #         raise rospy.exceptions.ROSException
    #     if len(base_link_bumper.states)>0:
    #         if base_link_bumper.states[0].collision2_name.split('::')[0]=='cylinder':
    #             target=True
    #         else:
    #             done=True
    #     return data,done,target



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
        if action == 0:
             vel_cmd.linear.x =0.5
        elif action == 1:
            vel_cmd.angular.z =0.2
            # vel_cmd.linear.x = 0.1
        elif action == 2:
            vel_cmd.angular.z =-0.2
            # vel_cmd.linear.x = 0.1
        else:
            vel_cmd.linear.x = -0.1

        self.vel_pub.publish(vel_cmd)
        data = None; base_link_bumper=None
        ifcollision=False;iftarget=False;ifflipped=False

        while (data and base_link_bumper) is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                base_link_bumper = rospy.wait_for_message('/turtlebot/base_link_bumper', ContactsState, timeout=5)
            except:
                pass
        if len(base_link_bumper.states) > 0:
            if base_link_bumper.states[0].collision2_name.split('::')[0] == 'cylinder':
                iftarget = True
            else:
                ifcollision = True


        targetPositionRe =self.getmodelstate
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            targetPositionRe=self.getmodelstate("cylinder","base_footprint")
        except(rospy.ServiceException)as e:
            print('get the target position failed !')


        angle,distance=self.getAngelDistance(targetPositionRe.pose.position.x,targetPositionRe.pose.position.y)




        anglereward = -np.round(abs(angle)/np.pi,2)
        distancereward = -np.round(distance / np.sqrt(200), 2)


            # Straight reward = 5, Max angle reward = 0.5
            #reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))

                # self.setRodomModelState('Target')
        if iftarget:
            goalreward = 200
            self.setRodomModelState('cylinder')
        else:
            goalreward=0

        if ifcollision:
            collisionreward = -200
        else:
            collisionreward =0

        if abs(angle)>0.5 :
            flippedreward=-1
            # ifflipped=True
        else:
            flippedreward=0

        done=ifcollision or ifflipped


        #print("reward is :"+str(reward))
        #print("distance2:"+str(distance2)+" distance1: "+str(distance1))

        # if abs(angle-euler[2])>np.pi:
        #     anglereward=-abs(angle-euler[2]-2*np.pi)/(np.pi)
        # else:
        #     anglereward=-abs(angle-euler[2])/(np.pi)
        # print("pose is: "+str(targetPositionRe.pose.position.x)+" "+str(targetPositionRe.pose.position.y))
        # print("anglereward is: "+str(anglereward)+"reward is :"+str(reward))
        reward=collisionreward+anglereward+goalreward+distancereward+flippedreward
        state = np.array(data.ranges) / np.sqrt(200)
        state=np.hstack([state,abs(distancereward),abs(anglereward)])
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
        try:
            data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            base_link_bumper = rospy.wait_for_message('/turtlebot/base_link_bumper', ContactsState, timeout=5)
        except:
            raise rospy.exceptions.ROSException

        state = np.array(data.ranges) / np.sqrt(200)
        done=False
        if len(base_link_bumper.states) > 0:
            if base_link_bumper.states[0].collision2_name.split('::')[0] == 'cylinder':
                target = True
            else:
                done = True
        # return data, done, target

        targetPositionRe = self.getmodelstate
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            targetPositionRe = self.getmodelstate("cylinder", "base_footprint")
        except(rospy.ServiceException)as e:
            print('get the target position failed !')
        angle, distance = self.getAngelDistance(targetPositionRe.pose.position.x, targetPositionRe.pose.position.y)



        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # state,done,target= self.calculate_observation(data)


        angler = np.round(angle/np.pi,2)
        distance=np.round(distance/np.sqrt(200),2)
        state=np.hstack([state,abs(distance),abs(angler)])

        # return np.asarray(state),done
        return state,done