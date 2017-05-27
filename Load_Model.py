########################################################
#               Written by: Yih Kai Teh                #
########################################################

import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import deque
env = gym.make('CartPole-v0')

discount = 0.99											# discount rate of the rewards
batch_size = 500										# size of mini-batch for training 
replay_size = 500000										# size of Experience Replay
maximum_episode_length = 200									# length of episodes (same as default setting)
replay_buffer = deque()										# store the details of experience in Experience Replay
test_episode = 10										# number of episodes for testing
number_hidden = 100										# number of units for the hidden layer
learning_rate = 0.0005										# learning rate for the optimizater 
state_dimension = env.observation_space.shape[0]						# number of observations which will be received
action_dimension = env.action_space.n								# number of actions which can be taken


# run the evaluation at every 20 episodes in order to monitor the performance during training 
def evaluation():
	global Total_Return, Total_Move, Total_Loss, Total_Episode
	episode_length = 0
	total_return = np.array([])

	# run evaluation for 10 times because each episode is stochastic
	for i in range(test_episode):
		observation = env.reset()

		# step through the environment until maximum episode length is reached (default is 200)
		for j in range(maximum_episode_length):

			# select best action using both network and step through the environment
			first_q_out = sess.run(Q1_output, feed_dict={state: [observation]})
			second_q_out = sess.run(Q2_output, feed_dict={state: [observation]})
			selected_action = np.argmax((first_q_out + second_q_out)[0])
			next_observation, reward, done, _ = env.step(selected_action)

			observation = next_observation

			if done:
				break

		# calculate the return (the -1 below is because the modified rewards of -1 at terminating step)
		total_return = np.append(total_return, (discount ** (j)) * -1)
		episode_length += j+1

	# display and store all the result for plotting the graph at the end of training
	average_episode_length = episode_length/test_episode
	print ('Mean Episode Length: %10f Mean Return: %10f' %(average_episode_length, np.mean(total_return)))


# initialize the weights and biases for all networks
weights = {	'first_Q_Learning_hidden_layer': tf.Variable(tf.truncated_normal([state_dimension, number_hidden])),
		'first_Q_Learning_output_layer': tf.Variable(tf.truncated_normal([number_hidden, action_dimension])),
		'second_Q_Learning_hidden_layer': tf.Variable(tf.truncated_normal([state_dimension, number_hidden])),
		'second_Q_Learning_output_layer': tf.Variable(tf.truncated_normal([number_hidden, action_dimension])),
		'stationary_target_hidden_layer': tf.Variable(tf.truncated_normal([state_dimension, number_hidden])),
		'stationary_target_output_layer': tf.Variable(tf.truncated_normal([number_hidden, action_dimension]))}

biases = {	'first_Q_Learning_hidden_layer': tf.Variable(tf.constant(0.01,shape = [number_hidden])),
		'first_Q_Learning_output_layer': tf.Variable(tf.constant(0.01,shape = [action_dimension])),
		'second_Q_Learning_hidden_layer': tf.Variable(tf.constant(0.01,shape = [number_hidden])),
		'second_Q_Learning_output_layer': tf.Variable(tf.constant(0.01,shape = [action_dimension])),
		'stationary_target_hidden_layer': tf.Variable(tf.constant(0.01,shape = [number_hidden])),
		'stationary_target_output_layer': tf.Variable(tf.constant(0.01,shape = [action_dimension]))}

# placeholder for state, actions and target value
state = tf.placeholder(tf.float32, shape=[None, state_dimension])
actions = tf.placeholder(tf.int32, shape=[None])
Q_target = tf.placeholder(tf.float32, shape=[None])

# first Q-learning network
Q1_hidden = tf.nn.relu(tf.matmul(state, weights['first_Q_Learning_hidden_layer']) + biases['first_Q_Learning_hidden_layer'])
Q1_output = tf.matmul(Q1_hidden, weights['first_Q_Learning_output_layer']) + biases['first_Q_Learning_output_layer']

# second Q-learning network
Q2_hidden = tf.nn.relu(tf.matmul(state, weights['second_Q_Learning_hidden_layer']) + biases['second_Q_Learning_hidden_layer'])
Q2_output = tf.matmul(Q2_hidden, weights['second_Q_Learning_output_layer']) + biases['second_Q_Learning_output_layer']

# stationary target network
stationary_hidden = tf.nn.relu(tf.matmul(state, weights['stationary_target_hidden_layer']) + biases['stationary_target_hidden_layer'])
stationary_output = tf.matmul(stationary_hidden, weights['stationary_target_output_layer']) + biases['stationary_target_output_layer']

# compute Q from current q_output and one hot actions
Q1 = tf.reduce_sum(tf.multiply(Q1_output, tf.one_hot(actions, action_dimension)), reduction_indices=1)
Q2 = tf.reduce_sum(tf.multiply(Q2_output, tf.one_hot(actions, action_dimension)), reduction_indices=1)

# loss operation 
loss_op_1 = tf.reduce_mean(tf.square(Q_target - Q1))
loss_op_2 = tf.reduce_mean(tf.square(Q_target - Q2))

# train operation
train_op_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_op_1)
train_op_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_op_2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver()
	saver.restore(sess, "./model_Double_Q-Learning/Double_Q-Learning")

	evaluation()


