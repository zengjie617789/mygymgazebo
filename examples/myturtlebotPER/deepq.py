#deep from navibot_ws
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as initializer
import matplotlib.pyplot as plt
import memory

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity,phi_length,batch_size):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.phi_length=phi_length
        self.n=batch_size
        self.States, self.actions, self.rewards, self.nextStates, self.terminals = np.empty((self.n, self.state_size)), \
        np.empty((self.n,)), np.empty(self.n, ), np.empty((self.n, self.state_size)), np.empty((self.n,))
    def add(self, p, state,action,reward,nextstate,terminal):
        tree_idx = self.data_pointer + self.capacity - 1
        # self.data[self.data_pointer] = data  # update data_frame
        self.States[self.data_pointer]=state
        self.actions[self.data_pointer]=action
        self.rewards[self.data_pointer]=reward
        self.nextStates[self.data_pointer]=nextstate
        self.terminals[self.data_pointer]=terminal

        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        # c=data_idx-self.phi_length + 3
        # perData=np.vstack((self.data[data_idx +self.phi_length -1],self.data[data_idx +self.phi_length -2],
        #                   self.data[data_idx+self.phi_length-3], self.data[data_idx]))
        return leaf_idx, self.tree[leaf_idx],data_idx


    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error


    def __init__(self, capacity,state_size,phi_length,minibathch):
        self.tree = SumTree(capacity,phi_length)
        self.state_size=state_size
        self.phi_length=phi_length
        self.n=minibathch
        self.number=0
        self.States, self.actions, self.rewards, self.nextStates, self.terminals = np.empty((self.n, self.state_size)), \
                                                                                   np.empty((self.n,)), np.empty(
            self.n, ), np.empty((self.n, self.state_size)), np.empty((self.n,))

    def store(self, state,action,reward,nextstate,terminal):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, state,action,reward,nextstate,terminal)   # set the max p for new p
        self.number+=1
    def sample(self,n,phi_length):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))

        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p,data_idx = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)

            self.States[i]=self.tree.States[data_idx:data_idx+4].reshape(1,self.state_size*self.phi_length)

            self.actions[i]=self.tree.actions[data_idx]
            self.rewards[i]=self.tree.rewards[data_idx]
            self.nextStates[i]=self.tree.nextStates[data_idx:data_idx+4].reshape(1,self.state_size*phi_length)
            self.terminals[i]=self.tree.terminals[data_idx]
            b_idx[i], b_memory[i, :] = idx,1
            # np.array([[self.data[data_idx - self.phi_length + 1]], [self.data[data_idx - self.phi_length + 2]]
            # [self.data[data_idx - self.phi_length + 3]], [self.data[data_idx]]])

        return b_idx, ISWeights,self.States,self.actions,self.rewards,self.nextStates,self.terminals

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    # def phi(self,state):
    #     indexs=np.arange(self.number-self.phi_length+1,self.number)
    #     phi=np.empty((self.phi_length,self.state_size))
    #     phi[0:self.phi_length-1]=self.States.take(indexs,axis=0)
    #     phi[-1]=state
    #     return phi.reshape(1,self.phi_length*self.state_size)
    def  phi(self,observation):
        state=np.zeros((self.phi_length,self.state_size))
        index=np.arange(self.number-self.phi_length+1,self.number)
        # for i in range(self.phi_length):
        state[0:self.phi_length]=self.tree.States.take(index)
        state[self.phi_length]=observation
        return state.reshape(1,self.state_size*self.phi_length)


class AgentTF:
    def __init__(self, state_size, action_size,phi_length, hidden_layers, batch_size, tau, gamma,learning_rate,memory_size,updatenetworks):

        self.state_size=state_size
        # self.phi_length=phi_length
        self.action_size=action_size
        self.hidden_layers=hidden_layers
        self.batch_size=batch_size
        self.tau=tau
        self.gamma=gamma
        self.lr=learning_rate
        self.memory_size=memory_size
        self.update_networks=updatenetworks
        self.phi_length=phi_length
        # tensorflow
        tf.reset_default_graph()

        self.mainQN = DQN("mainQN", self.state_size*self.phi_length, self.action_size,
                        self.hidden_layers, self.lr)
        self.targetQN = DQN("targetQN", self.state_size*self.phi_length, self.action_size,
                            self.hidden_layers, self.lr)

        self.trainables = tf.trainable_variables()
        self.target_ops = self.set_target_graph_vars(self.trainables, self.tau)
        self.saver = tf.train.Saver(max_to_keep=100)

        # self.memory = np.zeros((self.memory_size, state_size * 2 + 3))
        self.memory=Memory(self.memory_size,self.state_size,self.phi_length,self.batch_size)
        #START
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # Set the target network to be equal to the primary network
        self.update_target_graph(self.target_ops)
        self.writer=tf.summary.FileWriter("logDQN",self.sess.graph)
        self.merged=tf.summary.merge_all()
        self.learn_step_counter=0


    """ Auxiliary Methods """
    # Originally called updateTargetGraph
    def set_target_graph_vars(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []

        for idx,var in enumerate(tfVars[0:total_vars//2]): # Select the first half of the variables (mainQ net)
            op_holder.append( tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))

        return op_holder
    # Originally called updateTarget
    def update_target_graph(self, op_holder):
        for op in op_holder:
            self.sess.run(op)

    def getAction(self, state, epsilon):
        # State has to be np.array(44, 1)
        #if np.size(state) != (44,):
        #    raise ValueError
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.input:state})[0]
        return int(action)


    def addSample(self,state,action,reward,nextState,terminal):
        # transition=np.hstack((state,action,reward,nextState,terminal))
        # self.memory.store(transition)
        self.memory.store(state,action,reward,nextState,terminal)

    def train(self):
        # Train_batch = [s,a,r,s1,d]
        #Perform the Double-DQN update to the target Q-values
        tree_idx, ISWeights, self.States, self.actions, self.rewards, self.nextStates, self.terminals=\
            self.memory.sample(self.batch_size,self.phi_length)

        Q1 = self.sess.run(self.mainQN.predict,
                      feed_dict={self.mainQN.input:self.nextStates})

        Q2 = self.sess.run(self.targetQN.Qout,
                      feed_dict={self.targetQN.input:self.nextStates})

        end_multiplier = -(self.terminals - 1)
        doubleQ = Q2[range(self.batch_size),Q1]
        targetQ = self.rewards + (self.gamma*doubleQ*end_multiplier)

        # Update the network with our target values.
        _,td_error,loss = self.sess.run([self.mainQN.updateModel,self.mainQN.td_error, self.mainQN.loss],
                     feed_dict={self.mainQN.input:self.States,
                     self.mainQN.targetQ:targetQ,
                     self.mainQN.actions:self.actions,
                     self.mainQN.ISWeight:ISWeights})

        # Set the target network to be equal to the primary
        self.update_target_graph(self.target_ops)
        self.memory.batch_update(tree_idx,td_error)
        return loss

    def close(self):
        self.sess.close()


    def restore_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path+'/Checkpoints/')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print ('Loaded checkpoint: ', ckpt)

    def save_model(self, num_episode, path):
        self.saver.save(self.sess, path+'/Checkpoints/model-'+str(num_episode)+'.cptk')

    def store_transition(self, s, a, r, s_,done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_,done))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


class DQN():
    def __init__(self, net_name, state_size, action_size, hiddens, learning_rate):
        self.action_size=action_size
        self.net_name = net_name
        self.lr=learning_rate


        with tf.variable_scope(self.net_name):

            self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            self.ISWeight=tf.placeholder(tf.float32,[None,1],name='IS_weights')
            #self.input_state = tf.reshape(self.state, [-1, num_frames * state_size])

            # Weights of each layer
            self.W = {
                'W1': self.init_weight("W1", [state_size, hiddens[0]]),
                'W2': self.init_weight("W2", [hiddens[0], hiddens[1]]),
                'W3': self.init_weight("W3", [hiddens[1], hiddens[2]]),
                'W4': self.init_weight("W4", [hiddens[2], hiddens[3]]),
                'W5': self.init_weight("W5", [hiddens[3], hiddens[4]]),
                'AW': self.init_weight("AW", [hiddens[4]//2, action_size]),
                'VM': self.init_weight("VM", [hiddens[4]//2, 1])
            }

            self.b = {
                'b1': self.init_bias("b1", hiddens[0]),
                'b2': self.init_bias("b2", hiddens[1]),
                'b3': self.init_bias("b3", hiddens[2]),
                'b4': self.init_bias("b4", hiddens[3]),
                'b5': self.init_bias("b5", hiddens[4])
            }

            # Layers
            self.hidden1 = tf.nn.relu(tf.add(tf.matmul(self.input, self.W['W1']), self.b['b1']))
            self.hidden2 = tf.nn.relu(tf.add(tf.matmul(self.hidden1, self.W['W2']), self.b['b2']))
            self.hidden3 = tf.nn.relu(tf.add(tf.matmul(self.hidden2, self.W['W3']), self.b['b3']))
            self.hidden4 = tf.nn.relu(tf.add(tf.matmul(self.hidden3, self.W['W4']), self.b['b4']))
            self.hidden5 = tf.nn.relu(tf.add(tf.matmul(self.hidden4, self.W['W5']), self.b['b5']))

            # Compute the Advantage, Value, and total Q value
            self.A, self.V = tf.split(self.hidden5, 2, 1)
            self.Advantage = tf.matmul(self.A, self.W['AW'])
            self.Value = tf.matmul(self.V, self.W['VM'])
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))

            # Calcultate the action with highest Q value
            self.predict = tf.argmax(self.Qout, 1)

            # Compute the loss (sum of squared differences)
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_one_hot), axis=1)
            self.td_error = tf.square(self.targetQ - self.Q)
            # with tf.variable_scope("loss"):
            # self.loss = tf.reduce_mean(self.ISWeight*self.td_error)
            self.loss=tf.reduce_mean(self.ISWeight*self.td_error)
            tf.summary.scalar("loss",self.loss)

            self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.updateModel = self.trainer.minimize(self.loss)
            self.merged=tf.summary.merge_all()

    def init_weight(self, name, shape):
        return tf.get_variable(name=name, shape=shape, initializer=initializer.xavier_initializer(), dtype=tf.float32)

    def init_bias(self, name, shape):
        return tf.Variable(tf.random_normal([shape]))
        #initializer = tf.constant(np.random.rand(shape))
        #return tf.get_variable(name=name, initializer=initializer, dtype=tf.float32)