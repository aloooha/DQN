import gym
import numpy as np
import random
import tensorflow as tf
from DQN import DQN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


display = False  # switch for display

epsion = 0.2
epsion_min = 0.0001
alpha = 0.9

discount_factor = 1

batch_size = 32
action_op = np.array([[1, 0], [0, 1]])  # one_hot action

env = gym.make('CartPole-v0')

def max_next_Q(s_):
    """
    get max Q(s_, a*) by choice the best action a*
    :param s_:
    s_: state
    :return: Q(s_, a*)
    """
    if len(s_.shape) == 1:
        s_ = [s_]
    nextQ1 = net.predict_target_Q(s_, [action_op[0]]*s_.shape[0])
    nextQ2 = net.predict_target_Q(s_, [action_op[1]]*s_.shape[0])
    # print(nextQ1)
    # print(nextQ2)
    # print(np.max([nextQ1, nextQ2], axis=0))

    return np.max([nextQ1, nextQ2], axis=0)

def get_action(s):
    """
    return the action a* to gain max Q(s, a*) in state, with e-greedy to balance exploration and Exploitation
    :param
    s: the state now
    :return: action a, 1 or 2, following the policy
    """
    if len(s.shape) == 1:
        s = [s]
    if np.random.rand() < epsion:
        action = action_op[np.random.randint(0, 2)]
    else:
        q0 = net.predict(s, [action_op[0]])
        q1 = net.predict(s, [action_op[1]])
        action = action_op[0] if q0 > q1 else action_op[1]
    return np.argmax(action)



if __name__=="__main__":
    sess = tf.InteractiveSession()
    net = DQN(sess, s_dim=4, a_dim=2)
    Replay = []  # experience memory

    train_n = 0
    scores = []
    for n in range(100000):
        env.reset()
        a0 = env.action_space.sample()
        state, _, __, ___ = env.step(a0)  # get initial state
        i, done = 0, False
        if n % 10 == 0 and epsion > epsion_min:  # to decay exploration rate
            epsion = epsion*alpha
        while not done:
            i += 1
            a_ = get_action(state)
            if display:
                env.render()
            next_state, reward, done, _ = env.step(a_)
            Replay.append((state, action_op[a_], [reward], next_state, [done]))
            state = next_state
            # train every batch_size*5 transition record
            if Replay.__len__() > batch_size and Replay.__len__() % batch_size == 0:
                mini_batch = random.sample(Replay, batch_size)
                s, a, r, s_, done_batch = zip(*mini_batch)
                s = np.array(s)
                a = np.array(a)
                r = np.array(r)
                s_ = np.array(s_)
                done_batch = np.array(done_batch)
                # y = r+gamma*max(Q(s_,a*)) but if done y = r
                y_target = r + (1*~done_batch) * discount_factor * max_next_Q(s_)
                _, summary = net.train(s, a, y_target)
                train_n += 1
                net.writer.add_summary(summary, train_n)
        print("score: "+str(i))






