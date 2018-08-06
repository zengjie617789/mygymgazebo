"""This class stores all of the samples for training. It is able to
construct randomly selected batches of phi's from the stored history
"""

import numpy as np
import time
import tensorflow as tf


class DataSet(object):
    def __init__(self, stateSize, maxSteps, phiLength):
        '''Construct a DataSet
        Arguments:
        stateSize - number of parameters in state
        maxSteps - number of time steps to store
        phiLength - number of values to concatenate into a state
        rng - initialized np random number generator, used to
        choose random minibatches
        '''
        self.stateSize = stateSize
        self.maxSteps = maxSteps
        self.phiLength = phiLength
        self.rng = np.random.RandomState()

        # Allocate circular buffers and indices
        self.states = np.zeros((self.maxSteps, self.stateSize),
                               dtype=np.float32)
        self.actions = np.zeros(self.maxSteps, dtype='int32')
        self.rewards = np.zeros(self.maxSteps,
                                dtype=np.float32)
        self.nextStates = np.zeros((self.maxSteps, self.stateSize),
                                   dtype=np.float32)
        self.terminal = np.zeros(self.maxSteps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def addSample(self, state, action, reward, nextState, terminal):
        """ Add a  time step record
        Arguments:
            state - observed state
            action - action chosen by the agent
            reward - reward received after taking the action
            terminal - boolean inidcating whether the episode ended
            after this time step
        """
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.nextStates[self.top] = nextState
        self.terminal[self.top] = terminal

        if self.size == self.maxSteps:
            self.bottom = (self.bottom + 1) % self.maxSteps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.maxSteps

    def __len__(self):
        '''Return an approximate count for stored state transitions.'''
        return max(0, self.size - self.phiLength)

    def lastPhi(self):
        """Return the most recent phi (sequence of image frames)"""
        indexes = np.arange(self.top - self.phiLength, self.top)
        return self.states.take(indexes, axis=0, mode='wrap')

    def phi(self, state):
        """Return a phi (sequence of states), using the last phi length - 1,
        plus state.
        """
        indexes = np.arange(self.top - self.phiLength + 1, self.top)
        phi = np.empty((self.phiLength, self.stateSize),
                       dtype=np.float32)
        phi[0:self.phiLength - 1] = self.states.take(indexes, axis=0, mode='wrap')
        phi[-1] = state
        phi=phi.reshape(1,self.stateSize*self.phiLength)
        return phi

    def randomBatch(self, batchSize):
        """ Return cossreponding states, action, rewards, terminal status for
        batchSize. Randomly chosen state transitions
        """
        # Allocate space for the response
        states = np.zeros((batchSize,
                           self.phiLength,
                           self.stateSize),
                          dtype=np.float32)
        actions = np.zeros((batchSize, 1), dtype='int32')
        rewards = np.zeros((batchSize, 1), dtype=np.float32)
        nextStates = np.zeros((batchSize,
                               self.phiLength,
                               self.stateSize),
                              dtype=np.float32)
        terminal = np.zeros((batchSize, 1), dtype='bool')

        count = 0
        while count < batchSize:
            # Randomly choose a time step from the replay memory
            index = self.rng.randint(self.bottom,self.bottom + self.size - self.phiLength)
            # Both the before and after states contain phiLength
            # frames, overlapping except for the first and last
            allIndices = np.arange(index, index + self.phiLength)
            endIndex = index + self.phiLength - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            # if np.any(self.terminal.take(allIndices[0:-2], mode='wrap')):
            #    continue

            # Add the state transition to the response.
            states[count] = self.states.take(allIndices, axis=0, mode='wrap')
            actions[count] = self.actions.take(endIndex, mode='wrap')
            rewards[count] = self.rewards.take(endIndex, mode='wrap')
            nextStates[count] = self.nextStates.take(allIndices, axis=0, mode='wrap')
            terminal[count] = self.terminal.take(endIndex, mode='wrap')
            count += 1

        return states.reshape(batchSize, self.stateSize * self.phiLength), actions.reshape(
            batchSize, ), rewards.reshape(batchSize, ), nextStates.reshape(batchSize,
                                                                           self.stateSize * self.phiLength), terminal.reshape(
            batchSize, )


# TESTING CODE BELOW THIS POINT...




    # speed_tests()
    # # test_memory_usage_ok()
    # max_size_tests()
    # simple_tests()



