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
from examples.myturtlebot import deepq
import matplotlib.pyplot as plt
import rospy
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
    outdir = '/home/control511/KerasModel/gazebo_gym_experiments/'
    path = '/home/control511/KerasModel/turtle_c2_dqn_ep'
    plotter = liveplot.LivePlot(outdir)

    continue_execution = False
    # continue_execution=True

    epochs = 10000
    steps = 500
    updateTargetNetwork = 10000
    if continue_execution:
        explorationRate = 0.01
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
    # deepQ=DQNtf.DeepQNetwork(network_outputs, network_inputs, learningRate, discountFactor, e_greedy, updateTargetNetwork, memorySize, minibatch_size, output_graph=True)
    deepQ= deepq.AgentTF(network_inputs, network_outputs, network_structure, minibatch_size, 0.0001, discountFactor, learningRate, memorySize, updateTargetNetwork)
    if continue_execution:
    #     #Load weights, monitor info and parameter info.
    #     #ADD TRY CATCH fro this else
        deepQ.restore_model(outdir)

    env._max_episode_steps = steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()


    #start iterating from 'current epoch'.
    try:
        for epoch in range(current_epoch+1, epochs+1, 1):
            observation,done = env.reset()
            cumulated_reward = 0
            # done = False
            episode_step = 0

            # run until env returns done
            while not done:
                # env.render()
                # qValues = deepQ.getQValues(observation)
                #

                # action=deepQ.choose_action(observation,explorationRate)
                action=deepQ.getAction(observation,explorationRate)
                newObservation, reward, done, info = env.step(action)
                cumulated_reward += reward

                deepQ.store_transition(observation,action,reward,newObservation,done)


                if stepCounter >= learnStart:
                    deepQ.train()
                    # deepQ.learn()
                    # print("start to learn!")
                observation = newObservation

                if done:
                        m, s = divmod(int(time.time() - start_time), 60)
                        h, m = divmod(m, 60)
                        print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))

                stepCounter += 1
                episode_step += 1



            if stepCounter % 10000 == 0:
                # explorationRate *= 0.995  # epsilon decay
                explorationRate-=0.05
            # explorationRate -= (2.0/epochs)
                explorationRate = max(0.05, explorationRate)
            #     plotter.plot(env)
            #     plt.savefig('reward.png')
            #     deepQ.plot_cost()
            #     plt.savefig('loss.png')
                deepQ.save_model(epoch, outdir)
        deepQ.close()
    except rospy.exceptions.ROSException:

        deepQ.save_model(epoch,outdir)
        print("saved the model!")
        plt.close()
        env.close()
        deepQ.close()
