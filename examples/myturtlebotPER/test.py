import numpy as np
import time
from examples.myturtlebotPER import deepq
def simple_tests():
    print('...Starting Simple Test')
    np.random.seed(222)
    minibatch_size = 10  # 64 or 32 ?
    learnStart = 64
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 100
    network_inputs = 4
    network_outputs = 4
    network_structure = [100, 100, 100, 100, 100]
    # network_structure = [300,300]
    current_epoch = 0
    e_greedy = 0.9
    phi_length = 4
    dataset = deepq.AgentTF(network_inputs, network_outputs, phi_length, network_structure, minibatch_size, 0.0001,
                          discountFactor, learningRate, memorySize)
    state = np.random.random(4)*480
    action = np.random.randint(4)
    reward = np.random.random()
    terminal = False
    if np.random.random() < .05:
        terminal = True
    print('state', state)
    dataset.addSample(state, action, reward,state*2, terminal)
    print("S", dataset.memory.tree.States)
    print("A", dataset.memory.tree.actions)
    print("R", dataset.memory.tree.rewards)
    print("T", dataset.memory.tree.terminals)
    print("SIZE", dataset.memory.tree.capacity)
    print()
    print("LAST PHI", dataset.memory.lastPhi())
    print
    b_idx, ISWeights, States, actions, rewards, nextStates, terminals=dataset.memory.sample(6,4)

    print('States:', States)
    print('actions',actions)
    print('rewards',rewards)
    print('nextStates',nextStates)
    print('terminals',terminals)
    a=np.empty((2,))
    b=np.empty(2,)
    print(a)
    print(b)



def speed_tests():
    print('...Starting Speed Test')
    dataset = DataSet(stateSize=3,
                      maxSteps=20000, phiLength=4,
                      rng=np.random.RandomState(42))
    state = np.random.random(3)*480
    action = np.random.randint(3)
    reward = np.random.random()
    start = time.time()
    for i in range(100000):
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset.addSample(state, action, reward, terminal)
    print("samples per second: ", 100000 / (time.time() - start))

    start = time.time()
    for i in range(200):
        a = dataset.randomBatch(32)
    print("batches per second: ", 200 / (time.time() - start))

    print('Dataset.lastPhi(): ', dataset.lastPhi())

def main():
    simple_tests()
if __name__ == '__main__':
    main()


# LAST PHI [[6.94572226e-310 6.94572226e-310 6.94572226e-310 6.94572226e-310]
#  [6.94572226e-310 6.94572226e-310 6.94572226e-310 6.94572226e-310]
#  [6.94572226e-310 6.94572226e-310 0.00000000e+000 0.00000000e+000]
#  [2.52493645e+002 3.28952249e+002 4.01123610e+002 3.13807324e+002]]