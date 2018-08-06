#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import time
import os
import liveplot
from examples.myturtlebot import deepq,sample
import matplotlib.pyplot as plt
import rospy
import numpy as np
def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboMylabTurtlebotLidarNn-v0')
    # outdir = '/home/control511/KerasModel/gazebo_gym_experiments/'
    outdir ='/home/control511/KerasModel/DQNperModel/'
    path = '/home/control511/KerasModel/turtle_c2_dqn_ep'
    plotter = liveplot.LivePlot(outdir)

    # continue_execution = False
    continue_execution=True

    epochs = 10000
    steps = 500
    updateTargetNetwork = 10000
    if continue_execution:
        explorationRate = 1
    else:
        explorationRate = 1
    minibatch_size = 64
    learnStart = 64
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 1000000
    network_inputs = 10+2
    network_outputs = 4
    network_structure = [100,100,100,100,100]
    # network_structure = [300,300]
    current_epoch = 0
    e_greedy=0.9
    phi_length=4
    deepQ= deepq.AgentTF(network_inputs, network_outputs, phi_length, network_structure, minibatch_size, 0.0001, discountFactor, learningRate, memorySize, updateTargetNetwork)
    dataSet= sample.DataSet(network_inputs, memorySize, phi_length)
    memory=deepq.Memory(memorySize,state_size=network_inputs,phi_length=phi_length,minibathch=minibatch_size)
    if continue_execution:
    #     #Load weights, monitor info and parameter info.
    #     #ADD TRY CATCH fro this else
        deepQ.restore_model(outdir)

    env._max_episode_steps = steps # env returns done after _max_episode_steps
    # env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()
    cumulated_reward = 0
    done=False
    epoch=0
    cost_hist=[]
    reward_hist=[]
    try:
        for i in range(4):
            observation, done = env.reset()
            action=np.random.randint(4)
            newObservation,reward,done,info=env.step(action)
            # deepQ.addSample(observation,action,reward,newObservation,done)
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
                # phi=memory.phi(observation)
                phi=dataSet.phi(observation)
                # print(np.shape(phi))
                action=deepQ.getAction(phi,explorationRate)
                # action=deepQ.getAction(phi,explorationRate)
                newObservation, reward, done, info = env.step(action)
                cumulated_reward += reward

                # deepQ.store_transition(observation,action,reward,newObservation,done)
                # deepQ.addSample(observation,action,reward,newObservation,done)
                dataSet.addSample(observation, action, reward, newObservation, done)

                if stepCounter >= learnStart:
                    # batchStates, batchActions, batchRewards, batchNextStates, batchTerminals = \
                    #     dataSet.randomBatch(minibatch_size)


                    # loss = deepQ.train(batchStates, batchActions, batchRewards, batchNextStates, batchTerminals,stepCounter)
                    batchStates, batchActions, batchRewards, batchNextStates, batchTerminals = \
                        dataSet.randomBatch(minibatch_size)
                    loss=deepQ.train(batchStates,batchActions,batchRewards,batchNextStates,batchTerminals)
                    # deepQ.learn()
                    cost_hist.append(loss)
                observation = newObservation

                if done:
                        m, s = divmod(int(time.time() - start_time), 60)
                        h, m = divmod(m, 60)
                        print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                        reward_hist.append(cumulated_reward)
                stepCounter += 1
                episode_step += 1

                if stepCounter % 10000 == 0:
                    # explorationRate *= 0.995  # epsilon decay
                    explorationRate-=0.05
                # explorationRate -= (2.0/epochs)
                    explorationRate = max(0.05, explorationRate)
                    plt.figure(1)
                    plt.plot(np.arange(len(cost_hist)),cost_hist)
                    plt.ylabel('Cost')
                    plt.xlabel('ireation steps')
                    plt.savefig('loss.png')
                    plt.figure(2)
                    plt.plot(np.arange(len(reward_hist)),reward_hist)
                    plt.ylabel('cumulated_reward')
                    plt.xlabel('epochs')
                    plt.savefig('reward.png')
                    plt.show()
                #     deepQ.plot_cost()
                #
                    deepQ.save_model(epoch, outdir)
        deepQ.close()
    except rospy.exceptions.ROSException:

        deepQ.save_model(epoch,outdir)
        print("saved the model!")
        plt.close()
        env.close()
        deepQ.close()
