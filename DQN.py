import tensorflow as tf
import tensorlayer as tl
import numpy as np
import datetime

def variable_summaries(var):
    tf.summary.scalar('mean', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class DQN(object):
    def __init__(self, sess, s_dim, a_dim=2):
        self.sess = sess
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.state = tf.placeholder(tf.float32, shape=[None, s_dim], name="state")
        self.action = tf.placeholder(tf.float32, shape=[None, a_dim], name="action")
        self.target_Q = tf.placeholder(tf.float32, shape=[None, 1], name="target_Q")

        self.pred_Q, self.network_all_params = self.creat_network()

        self.pred_target_Q, self.target_all_params = self.creat_target_network()  # target network

        self.loss = tl.cost.mean_squared_error(self.pred_Q, self.target_Q)
        self.global_steps = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.015, self.global_steps, 50, 0.96, staircase=False)
        self.C_step = 0

        self.grads = tf.gradients(self.loss, self.network_all_params)
        self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
            apply_gradients(zip(self.grads, self.network_all_params), global_step=self.global_steps)

        tl.layers.initialize_global_variables(self.sess)

        variable_summaries(self.loss)
        variable_summaries(self.learning_rate)
        self.merged = tf.summary.merge_all()
        nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        print(nowTime)
        self.writer = tf.summary.FileWriter("./result/tensorboard/"+nowTime, sess.graph)


    def creat_target_network(self):
        input_s = tf.concat([self.state, self.action], axis=1)
        network = tl.layers.InputLayer(input_s)  # 6
        network = tl.layers.DenseLayer(network, n_units=32, act=tf.nn.relu, name="tfc1")  # 32
        network = tl.layers.DenseLayer(network, n_units=32, act=tf.nn.relu, name="tfc2")  # 32
        network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name="tfc3")  # 1
        network_all_params = network.all_params
        return network.outputs,  network_all_params

    def creat_network(self):
        input_s = tf.concat([self.state, self.action], axis=1)
        network = tl.layers.InputLayer(input_s)  # 6
        network = tl.layers.DenseLayer(network, n_units=32, act=tf.nn.relu, name="fc1")  # 32
        network = tl.layers.DenseLayer(network, n_units=32, act=tf.nn.relu, name="fc2")  # 32
        network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name="fc3")  # 1
        network_all_params = network.all_params
        return network.outputs, network_all_params

    def predict(self, state, action):
        pred_Q = self.sess.run(self.pred_Q, feed_dict={self.state: state,
                                                       self.action: action})
        return pred_Q

    def predict_target_Q(self, state, action):
        pred_target_Q = self.sess.run(self.pred_target_Q, feed_dict={self.state: state, self.action: action})
        return pred_target_Q


    def train(self, state, action, target_Q):
        self.C_step += 1
        _, summary = self.sess.run([self.optimize, self.merged], feed_dict={self.state: state,
                                                                            self.action: action,
                                                                            self.target_Q: target_Q})
        # print("update network")
        if self.C_step % 10 == 0:  # reset target network
            for t, p in zip(self.target_all_params, self.network_all_params):
                self.sess.run(tf.assign(t, p))

        return _, summary









