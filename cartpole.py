import time 
import gym
import numpy as np
import tensorflow as tf 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gamma',  type=float, default=0.99, help='a discount factor')
parser.add_argument('--num_iter', type=int, default=10000, help='maximum number of iterations')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')

args = parser.parse_args()

gamma 		= args.gamma
num_iter 	= args.num_iter
l_rate 		= args.learning_rate

input_dim 	= 4
hidden_dim 	= 2

################################################ Tensorflow graph ###################################################

#weight matrices
W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dim], minval=-0.01, maxval=0.01), dtype=tf.float32, name='W1')
W2 = tf.Variable(tf.random_uniform([hidden_dim, 1], minval=-0.01, maxval=0.01), dtype=tf.float32, name='W2')

s = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
r = tf.placeholder(dtype=tf.float32, shape=[None, 1])

#2-layer perceptron
h1 = tf.nn.tanh(tf.matmul(s, W1))
out = tf.matmul(h1, W2)
prob = tf.nn.sigmoid(out)
#cross entropy loss
loss = - tf.reduce_mean((y*tf.log(1e-9 + prob) + (1-y)*(tf.log(1 + 1e-9 - prob)))*r, 0)

opt = tf.train.AdamOptimizer(learning_rate=l_rate)
#compute the gradients and apply them (backprop)
grads_and_vars = opt.compute_gradients(loss)
train_step = opt.apply_gradients(grads_and_vars)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#####################################################################################################################

def discounted_rewards(rewards, gamma):
	#discount rewards
	return np.array([sum([gamma**t*r for t, r in enumerate(rewards[i:])]) for i in range(len(rewards))])

accum_rewards = np.zeros(100)

env = gym.make('CartPole-v0')

for episode in range(num_iter):

	labels, rewards, states = [], [], []

	obs = env.reset()

	while True:
		
		env.render()

		states.append(obs)

		p = float(sess.run(prob, feed_dict = {s: np.reshape(obs, [1, input_dim])}))
		action = int(np.random.choice(2, 1, p = [1-p, p]))
		
		obs, reward, done, info = env.step(action)

		labels.append(action)
		rewards.append(reward)

		if done:

			epr = np.vstack(discounted_rewards(rewards, gamma))
			eps = np.vstack(states)
			epl = np.vstack(labels)

			epr -= np.mean(epr)
			sess.run(train_step, feed_dict = {s: eps, y: epl, r: epr})

			accum_rewards[:-1] = accum_rewards[1:]
			accum_rewards[-1] = np.sum(rewards)

			#average reward (number of timesteps) over the last 100 steps 
			print('Running average steps:', np.mean(accum_rewards[accum_rewards > 0]), 'Episode:', episode+1)
			
			break