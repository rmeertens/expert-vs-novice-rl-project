#!/usr/bin/env python
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import gym
import random

env = gym.make('MountainCar-v0')
observation = env.reset()

## Save the amount of input and output variables
OBSERVATION_SPACE_LEN = len(observation)
ACTION_SPACE_LEN = env.action_space.n

## Global variables needed for human control
human_agent_action = 0
human_gives_input = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    """ If a human pressed a key we record if he gives a reward or punishment"""
    global human_agent_action, human_wants_restart, human_sets_pause, human_gives_input

    ## pushing the space bar pauses the game
    if key==32: human_sets_pause = not human_sets_pause

    a = int( key - ord('0') )
    print("Pressed: " + str(a))
    if a==0:
        human_wants_restart = True
    if a==1:
        human_agent_action = 1
        human_gives_input = True
    elif a==2:
        human_agent_action = -1
        human_gives_input = True

def key_release(key, mod):
    """ On key release: stop giving reward or punishment"""
    global human_agent_action, human_gives_input
    a = int( key - ord('0') )
    if a==1 or a==2:
        human_agent_action = 0
        human_gives_input = False

## Attach the functions that must be called when we press a key
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def build_neural_network():

    # Network input
    networkstate = tf.placeholder(tf.float32, [None, OBSERVATION_SPACE_LEN], name="input")
    networkaction = tf.placeholder(tf.int32, [None], name="actioninput")
    networkreward = tf.placeholder(tf.float32,[None], name="groundtruth_reward")
    action_onehot = tf.one_hot(networkaction, ACTION_SPACE_LEN, name="actiononehot")

    # The variable in our network:
    w1 = tf.Variable(tf.random_normal([OBSERVATION_SPACE_LEN,16], stddev=0.35), name="W1")
    w2 = tf.Variable(tf.random_normal([16,32], stddev=0.35), name="W2")
    w3 = tf.Variable(tf.random_normal([32,8], stddev=0.35), name="W3")
    w4 = tf.Variable(tf.random_normal([8,ACTION_SPACE_LEN], stddev=0.35), name="W4")
    b1 = tf.Variable(tf.zeros([16]), name="B1")
    b2 = tf.Variable(tf.zeros([32]), name="B2")
    b3 = tf.Variable(tf.zeros([8]), name="B3")
    b4 = tf.Variable(tf.zeros(ACTION_SPACE_LEN), name="B4")

    # The network layout
    layer1 = tf.nn.relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,w2), b2), name="Result2")
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2,w3), b3), name="Result3")
    predictedreward = tf.add(tf.matmul(layer3,w4), b4, name="predictedReward")

    # Learning
    qreward = tf.reduce_sum(tf.multiply(predictedreward, action_onehot), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(networkreward - qreward))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)
    merged_summary = tf.summary.merge_all()
    return networkstate, networkaction, networkreward, predictedreward, loss, optimizer, merged_summary


networkstate, networkaction, networkreward, predictedreward, loss, optimizer, merged_summary = build_neural_network()
sess = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter('trainsummary',sess.graph)
sess.run(tf.global_variables_initializer())

replay_memory = []  # (state, action, reward, terminalstate, state_t+1)
epsilon = 1.0
BATCH_SIZE = 32
GAMMA = 0.9
MAX_LEN_REPLAY_MEMORY = 30000
FRAMES_TO_PLAY = 300001
MIN_FRAMES_FOR_LEARNING = 1000
summary = None

epsilon = 1.0
losshere = 0.0
for i_epoch in range(FRAMES_TO_PLAY):

    ### Select an action and perform this
    if random.random() > epsilon:
        predreward = sess.run(predictedreward, feed_dict={networkstate: [observation]})
        action = np.argmax(np.array(predreward[0]))
    else:
        action = env.action_space.sample()

    print("Taking action: " + str(action))
    newobservation, reward, terminal, info = env.step(action)

    env.render()
    ### Override the reward given by the game with the reward given by the human
    if human_gives_input:
        reward = human_agent_action
    else:
        reward = 0

    ### Add the observation to our replay memory
    replay_memory.append((observation, action, reward, terminal, newobservation))

    ### Reset the environment if the agent died
    if human_wants_restart:
        newobservation = env.reset()
        human_wants_restart = False
    observation = newobservation

    ### Learn once we have enough frames to start learning
    if len(replay_memory) > MIN_FRAMES_FOR_LEARNING:
        experiences = random.sample(replay_memory, BATCH_SIZE)
        totrain = []  # (state, action, delayed_reward)

        ### Calculate the predicted reward
        nextstates = [var[4] for var in experiences]
        pred_reward = sess.run(predictedreward, feed_dict={networkstate: nextstates})

        ### Set the "ground truth": the value our network has to predict:
        for index in range(BATCH_SIZE):
            state, action, reward, terminalstate, newstate = experiences[index]
            predicted_reward = max(pred_reward[index])

            if terminalstate:
                delayedreward = reward
            else:
                delayedreward = reward + GAMMA * predicted_reward
            totrain.append((state, action, delayedreward))

        ### Feed the train batch to the algorithm
        states = [var[0] for var in totrain]
        actions = [var[1] for var in totrain]
        rewards = [var[2] for var in totrain]
        _, l, summary = sess.run([optimizer, loss, merged_summary],
                                 feed_dict={networkstate: states, networkaction: actions, networkreward: rewards})
        losshere += l
        ### If our memory is too big: remove the first element
        if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
            replay_memory = replay_memory[1:]

        ### Print the progress in terminal
        if i_epoch % 100 == 1:
            summary_writer.add_summary(summary, i_epoch)
        if i_epoch % 1000 == 1:
            print("Epoch %d, loss: %f, epsilon: %f" % (i_epoch, losshere, epsilon))
            losshere = 0.0

    epsilon -= 0.001
    if epsilon < 0.1:
        epsilon = 0.1

    time.sleep(0.1)

    while human_sets_pause:
        env.render()
        import time
        time.sleep(0.3333)
