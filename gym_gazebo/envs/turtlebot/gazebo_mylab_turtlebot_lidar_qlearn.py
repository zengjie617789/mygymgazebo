import gym
import rospy
import roslaunch
import time
import numpy as np
import math
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import tf
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan

from gym.utils import seeding
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelStates
import cmath


class GazeboMylabTurtlebotLidarEnv(gazebo_env.GazeboEnv):
    store = [0]
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboMylabTurtlebotLidar_v0.launch")
        #gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.getmodelstate=rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.distance=0
        self.angle=0
        self._seed()

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.3
            self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        robot_tf=None
        while robot_tf is None:
            try:
                robot_tf = rospy.wait_for_message("/tf", TFMessage, timeout=5)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data,5)

        targetPosition=ModelStates()
        mobilePosition=ModelStates()
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            targetPosition=self.getmodelstate('Target', 'world')
            mobilePosition=self.getmodelstate('mobile_base', 'world')
        except(rospy.ServiceException)as e:
            print('get the target position failed !')



        # distance2 = math.hypot(mobilePosition.pose.position.x - targetPosition.pose.position.x,
        #                        mobilePosition.pose.position.y - targetPosition.pose.position.y)
        x= targetPosition.pose.position.x-mobilePosition.pose.position.x
        y= targetPosition.pose.position.y-mobilePosition.pose.position.y
        distance,angle= cmath.polar(complex(x,y))
        if x < 0 and y < 0:
            angle = -np.pi - np.arcsin(y / distance)
        elif x < 0 and y > 0:
            angle = np.pi - np.arcsin(y / distance)
        else:
            angle = np.arcsin(y / distance)

        quaternion=(robot_tf.transforms[0].transform.rotation.x,
                    robot_tf.transforms[0].transform.rotation.y,
                    robot_tf.transforms[0].transform.rotation.z,
                    robot_tf.transforms[0].transform.rotation.w)
        euler=np.array(tf.transformations.euler_from_quaternion(quaternion))

        if distance<=0.3:
            target=True
            # print("targetPosition.pose.position.y"+str(targetPosition.pose.position.y))
            # print("mobilePosition.pose.position.y"+str(mobilePosition.pose.position.y))
        else:
            target=False

        #target=False
        if not done:
            if target:
                reward = 200
                done = True
            else:
                distance1=GazeboMylabTurtlebotLidarEnv.store[0]
                #reward = round(10*(distance1-distance2),2)
                reward=-distance
                #GazeboMylabTurtlebotLidarEnv.store[0]=distance2
                # print('distance: '+str(distance1)+'  distance2: '+str(distance2)+' store: '+str(GazeboMylabTurtlebotLidarEnv.store[0]))
        else:
            reward = -200

        # if abs(angle-euler[2])>math.pi:
        #     anglereward=-abs(-angle-euler[2])
        # else:
        #     anglereward=-abs(angle-euler[2])

        if angle-euler[2]< np.pi/2 or angle-euler[2]>np.pi*3/2:
            anglereward=-(max(angle,euler[2])-min(angle,euler[2]))
        else:
            anglereward=-np.pi
        #print(anglereward,reward)
        reward=reward+anglereward
        self.distance=np.round(distance,1)
        self.angle=np.round(angle,1)
        state.extend([self.distance,self.angle])

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

        state , done= self.discretize_observation(data,5)
        #state=np.hstack([state,self.distance,self.angle])
        state.extend([self.distance,self.angle])
        print(state)
        return state
