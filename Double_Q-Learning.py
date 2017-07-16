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

seed = 600
tf.set_random_seed(seed)
env.seed(seed)
np.random.seed(seed)
random.seed(seed)

discount                = 0.99								  # discount rate of the rewards
batch_size              = 500								  # size of mini-batch for training 
replay_size             = 500000							  # size of Experience Replay
maximum_episode_length  = 200								  # length of episodes (same as default setting)
replay_buffer           = deque()							  # store the details of experience in Experience Replay
test_episode            = 10								  # number of episodes for testing
total_episode           = 500								  # number of episodes for training
number_hidden           = 100								  # number of units for the hidden layer
learning_rate           = 0.0005							  # learning rate for the optimizater 
state_dimension         = env.observation_space.shape[0]				  # number of observations which will be received
action_dimension        = env.action_space.n						  # number of actions which can be taken
Total_Move = Total_Loss = Total_Return = Total_Episode = cummulative_loss = np.array([])  # store the details for plotting the graph


# epsilon greedy with fix epsilon. A decay epsilon can be implemented to reach optimal policy.
def epsilon_greedy(obs):
	if float(np.random.rand(1)) <= 0.05:
		action = env.action_space.sample()	# random action
	else:
		Q1_out = sess.run(Q1_output, feed_dict={state: [obs]})
		Q2_out = sess.run(Q2_output, feed_dict={state: [obs]})

		action = np.argmax((Q1_out + Q2_out)[0]) # optimal action
	return action


# store the transitions and details information into the experience replay
def store_all_information(obs, action, reward, next_obs, done):
	global replay_buffer
	replay_buffer.append((obs, action, reward, next_obs, done)) # state, action, reward, next_state, continue

	# clear the experience replay by one to allow for new memory when it is full 
	if len(replay_buffer) > replay_size:
		replay_buffer.popleft()

	# start training when experience replay has enough experience for one batch size
	if len(replay_buffer) > batch_size:
		loss = train_Q_network()
		return loss
	else:
		return 0


# randomly sample experience from experience replay for training
def train_Q_network():
	global replay_buffer

	# randomly sample experience
	minibatch        = random.sample(replay_buffer, batch_size)
	batch_state      = [data[0] for data in minibatch]
	batch_action     = [data[1] for data in minibatch]
	batch_reward     = [data[2] for data in minibatch]
	batch_next_state = [data[3] for data in minibatch]

	# randomly update either first or second Q network
	if np.random.rand(1) > 0.5:
		batch_Q1_target = []
		batch_Q1_value  = stationary_output.eval(feed_dict={state: batch_next_state})
		for i in range(0, batch_size):
			continues = minibatch[i][4]
			batch_Q1_target.append(batch_reward[i] + discount * continues * np.max(batch_Q1_value[i]))

		cost, _ = sess.run([loss_op_1, train_op_1], feed_dict={state: batch_state, actions: batch_action, Q_target: batch_Q1_target})
	else:
		batch_Q2_target = []
		batch_Q2_value  = stationary_output.eval(feed_dict={state: batch_next_state})
		for i in range(0, batch_size):
			continues = minibatch[i][4]
			batch_Q2_target.append(batch_reward[i] + discount * continues * np.max(batch_Q2_value[i]))

		cost, _ = sess.run([loss_op_2, train_op_2], feed_dict={state: batch_state, actions: batch_action, Q_target: batch_Q2_target})

	return cost


# run the evaluation at every 20 episodes in order to monitor the performance during training 
def evaluation(episode, cummulative_loss):
	global Total_Return, Total_Move, Total_Loss, Total_Episode
	episode_length = 0
	total_return   = np.array([])

	# run evaluation for 10 times because each episode is stochastic
	for i in range(test_episode):
		observation = env.reset()

		# step through the environment until maximum episode length is reached (default is 200)
		for j in range(maximum_episode_length):

			# select best action using both network and step through the environment
			first_q_out                       = sess.run(Q1_output, feed_dict={state: [observation]})
			second_q_out                      = sess.run(Q2_output, feed_dict={state: [observation]})
			selected_action                   = np.argmax((first_q_out + second_q_out)[0])
			next_observation, reward, done, _ = env.step(selected_action)

			observation = next_observation

			if done:
				break

		# calculate the return (the -1 below is because the modified rewards of -1 at terminating step)
		total_return    = np.append(total_return, (discount ** (j)) * -1)
		episode_length += j+1

	# display and store all the result for plotting the graph at the end of training
	average_episode_length = episode_length/test_episode
	print ('Episode: %4d Mean Episode Length: %10f Mean Return: %10f' %(episode + 1, average_episode_length, np.mean(total_return)))
	Total_Return  = np.append(Total_Return, np.mean(total_return))
	Total_Move    = np.append(Total_Move, average_episode_length)
	Total_Loss    = np.append(Total_Loss, np.mean(cummulative_loss))
	Total_Episode = np.append(Total_Episode, episode + 1)


# initialize the weights and biases for all networks
weights = {	'first_Q_Learning_hidden_layer' : tf.Variable(tf.truncated_normal([state_dimension, number_hidden])),
		'first_Q_Learning_output_layer' : tf.Variable(tf.truncated_normal([number_hidden, action_dimension])),
		'second_Q_Learning_hidden_layer': tf.Variable(tf.truncated_normal([state_dimension, number_hidden])),
		'second_Q_Learning_output_layer': tf.Variable(tf.truncated_normal([number_hidden, action_dimension])),
		'stationary_target_hidden_layer': tf.Variable(tf.truncated_normal([state_dimension, number_hidden])),
		'stationary_target_output_layer': tf.Variable(tf.truncated_normal([number_hidden, action_dimension]))}

biases  = {	'first_Q_Learning_hidden_layer' : tf.Variable(tf.constant(0.01,shape = [number_hidden])),
		'first_Q_Learning_output_layer' : tf.Variable(tf.constant(0.01,shape = [action_dimension])),
		'second_Q_Learning_hidden_layer': tf.Variable(tf.constant(0.01,shape = [number_hidden])),
		'second_Q_Learning_output_layer': tf.Variable(tf.constant(0.01,shape = [action_dimension])),
		'stationary_target_hidden_layer': tf.Variable(tf.constant(0.01,shape = [number_hidden])),
		'stationary_target_output_layer': tf.Variable(tf.constant(0.01,shape = [action_dimension]))}

# update weights and biases of First Q-Learning
update_Q1_weight_hidden = weights['stationary_target_hidden_layer'].assign(weights['first_Q_Learning_hidden_layer'])
update_Q1_weight_output = weights['stationary_target_output_layer'].assign(weights['first_Q_Learning_output_layer'])
update_Q1_bias_hidden   = biases['stationary_target_hidden_layer'].assign(biases['first_Q_Learning_hidden_layer'])
update_Q1_bias_output   = biases['stationary_target_output_layer'].assign(biases['first_Q_Learning_output_layer'])
update_all_Q1           = [update_Q1_weight_hidden, update_Q1_weight_output, update_Q1_bias_hidden, update_Q1_bias_output]

# update weights and biases of Second Q-Learning
update_Q2_weight_hidden = weights['stationary_target_hidden_layer'].assign(weights['second_Q_Learning_hidden_layer'])
update_Q2_weight_output = weights['stationary_target_output_layer'].assign(weights['second_Q_Learning_output_layer'])
update_Q2_bias_hidden   = biases['stationary_target_hidden_layer'].assign(biases['second_Q_Learning_hidden_layer'])
update_Q2_bias_output   = biases['stationary_target_output_layer'].assign(biases['second_Q_Learning_output_layer'])
update_all_Q2           = [update_Q2_weight_hidden, update_Q2_weight_output, update_Q2_bias_hidden, update_Q2_bias_output]

# placeholder for state, actions and target value
state     = tf.placeholder(tf.float32, shape=[None, state_dimension])
actions   = tf.placeholder(tf.int32, shape=[None])
Q_target  = tf.placeholder(tf.float32, shape=[None])

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

	# train for 1000 episodes
	for episode in range(total_episode):
		observation = env.reset()

		# updates the stationary network every 5 episodes
		if episode % 5 == 0:
			sess.run(update_all_Q1)
			sess.run(update_all_Q2)

		# step through the environment until maximum episode length is reached (default is 200)
		for t in range(maximum_episode_length):

			# select action based on epsilon greedy and use that action in the environment
			selected_action                   = epsilon_greedy(observation)
			next_observation, reward, done, _ = env.step(selected_action)

			# modifies the reward into sparse rewards
			reward = -1 if done else 0

			# store all the transition and details into experience replay
			cost = store_all_information(observation, selected_action, reward, next_observation, 1.0 - done)

			# start accumulating the loss after experience replay has enough experience for one batch size to train
			if len(replay_buffer) > batch_size:
				cummulative_loss = np.append(cummulative_loss, cost)

			observation = next_observation

			if done:
				break

		# run evaluation every 20 episodes
		if (episode + 1) % 20 == 0:
			evaluation(episode, cummulative_loss)

	print("Average Episode Length: %f" % (np.mean(Total_Move)))
	print("Average Return: %f" % (np.mean(Total_Return)))

	# Save the model
	saver.save(sess, './model_Double_Q-Learning/Double_Q-Learning')

	# Plot for the graph for the return, episodes length and loss throughout the training
	plt.figure(0)
	plt.plot(Total_Episode, Total_Return)
	plt.title('Plot of Performance Over %d Episodes' %(total_episode))
	plt.xlabel('Episodes')
	plt.ylabel('Performance (Mean Discounted Return)')
	plt.savefig('Plot of Performance Over %d Episodes.png' %(total_episode))

	plt.figure(1)
	plt.plot(Total_Episode, Total_Move)
	plt.title('Plot of Performance Over %d Episodes' %(total_episode))
	plt.xlabel('Episodes')
	plt.ylabel('Performance (Episode Length)')
	plt.savefig('Plot of Performance (Episode Length) Over %d Episodes.png' %(total_episode))

	plt.figure(2)
	plt.plot(Total_Episode, Total_Loss)
	plt.title('Plot of Loss Over %d Episodes' %(total_episode))
	plt.xlabel('Episodes')
	plt.ylabel('Training Loss')
	plt.savefig('Plot of Loss Over %d Episodes.png' %(total_episode))
