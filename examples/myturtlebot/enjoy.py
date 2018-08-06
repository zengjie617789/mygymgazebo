import gym
import time
import os
import liveplot
from examples.myturtlebot import deepq, sample
import matplotlib.pyplot as plt
import rospy
import numpy as np

def main():
    minibatch_size = 64
    learnStart = 64
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 1000000
    network_inputs = 10 + 2
    network_outputs = 4
    network_structure = [100, 100, 100, 100, 100]
    # network_structure = [300,300]
    current_epoch = 0
    e_greedy = 0.9
    phi_length = 4
    steps = 500
    epoch=0
    epochs=10000
    stepsCouter=0
    explorationRate=0.05
    updateTargetNetwork = 10000
    dataSet=sample.Data_set(network_inputs,memorySize,phi_length)

    outdir = '/home/control511/KerasModel/gazebo_gym_experiments/'
    env = gym.make('GazeboMylabTurtlebotLidarNn-v0')
    deepQ= deepq.AgentTF(network_inputs, network_outputs, phi_length, network_structure, minibatch_size, 0.0001, discountFactor, learningRate, memorySize, updateTargetNetwork)
    deepQ.restore_model(outdir)

    env._max_episode_steps = steps

    try:
        for i in range(4):
            observation, done = env.reset()
            action=np.random.randint(4)
            newObservation,reward,done,info=env.step(action)
            dataSet.addSample(observation,action,reward,newObservation,done)



    #start iterating from 'current epoch'.

        for epoch in range(current_epoch+1, epochs+1, 1):
            observation,done = env.reset()
            done = False
            cumulated_reward=0
            episode_step = 0
            # run until env returns done
            while not done:
                # env.render()
                # qValues = deepQ.getQValues(observation)
                #

                # action=deepQ.choose_action(observation,explorationRate)
                phi=dataSet.phi(observation)
                action=deepQ.getAction(phi,explorationRate)
                newObservation, reward, done, info = env.step(action)
                cumulated_reward += reward
                stepsCouter+=1
                episode_step+=1
        deepQ.close()
    except rospy.exceptions.ROSException:
        print("load failed! ")

if __name__=="__main__":
    main()



