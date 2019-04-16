import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from AAPI import *
from utils import *
import pickle
import json
import numpy as np
import tensorflow as tf
from copy import deepcopy

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config) 


class RLController(object):
	"""
	Controller class
	"""
	def __init__(self, env, controller_name='controller'):

		# predefined constants for control types
		self.PHASE_SELECTION = 1
		self.TIME_EXTENSION = 2
		self.FIXED_TIMES = 3

		# predefined constants for coordination types
		self.INDEP = 1
		self.JOINT = 2

		self.controller_name = controller_name
		self.CONTROL_TYPE = env.CONTROL_TYPE
		self.env = env # RL environment object

		# statistics of the RL algorthm
		self.ep_rewards = {junct_id:[] for junct_id in self.env.JUNCTIONS} if self.env.COORDINATION_TYPE == self.env.INDEP else {0:[]}
		self.ep_losses = {junct_id:[] for junct_id in self.env.JUNCTIONS} if self.env.COORDINATION_TYPE == self.env.INDEP else {0:[]}
		self.ep_nn_losses = {junct_id:[] for junct_id in self.env.JUNCTIONS} if self.env.COORDINATION_TYPE == self.env.INDEP else {0:[]}
		self.ep_maes = {junct_id:[] for junct_id in self.env.JUNCTIONS} if self.env.COORDINATION_TYPE == self.env.INDEP else {0:[]}
		self.ep_qs = {junct_id:[] for junct_id in self.env.JUNCTIONS} if self.env.COORDINATION_TYPE == self.env.INDEP else {0:[]}
		self.ep_trajectory = {junct_id:[] for junct_id in self.env.JUNCTIONS}
		self.rewards = []
		self.losses = []
		self.nn_losses = []
		self.maes = []
		self.qs = []

		# statistics of the environment
		self.ep_env_stats = {
			'total_travel_time': [],
			'veh_waiting': [],
			'veh_count': [],
			'avg_travel_time': [],
			'avg_delay': [],
			'avg_stop_time': [],
			'avg_queue_length': [],
		}


	def initialize(self):
		log('[INIT] Initializing RL controller...', level=2)
		self.last_env_inputs = {junct_id:None for junct_id in self.env.JUNCTIONS} if self.env.COORDINATION_TYPE == self.env.INDEP else {0:None}
		
		# initialize RL agents
		self.build_rl_agents(self.env.test_mode)
		for junct_id in self.env.JUNCTIONS if self.env.COORDINATION_TYPE == self.env.INDEP else [0]:
			self.rl_agents[junct_id].training = not self.env.test_mode
			self.rl_agents[junct_id].ep_step = 0

		# load current agent checkpoint from file (if exists)
		self.load_agents()


	def build_rl_agents(self, test_mode):
		log('[INIT] Building RL agents...', level=2)
		self.models = {}
		self.rl_agents = {}

		for junct_id in self.env.JUNCTIONS if self.env.COORDINATION_TYPE == self.env.INDEP else [0]:
			# build neural network
			model = Sequential()
			model.add(Flatten(input_shape=(1,10+4)))
			model.add(Dense(12))
			model.add(Activation('relu'))
			model.add(Dense(12))
			model.add(Activation('relu'))
			model.add(Dropout(0.5))
			model.add(Dense(9, activation='linear',
	                kernel_regularizer=regularizers.l2(0.05),
	                #activity_regularizer=regularizers.l1(0.01)
			))
			self.models[junct_id] = model

			# build rl_agent
			memory = SequentialMemory(limit=10000, window_length=1)
			policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=.1, value_test=.05, nb_steps=500)
			rl_agent = DQNAgent(model=model, nb_actions=9, memory=memory, nb_steps_warmup=0,
			               enable_dueling_network=True, dueling_type='avg', target_model_update=0.01, policy=policy)
			rl_agent.compile(Adam(lr=1e-3), metrics=['mae'])
			self.rl_agents[junct_id] = rl_agent

		raise NotImplementedError


	def control(self, sim_time, timeSta, time,  acycle, junct_id, act=True):
		log('[CONTROL] ------------------------------------------------------------------------', level=2)
		log('[CONTROL] Control called for junction %d! Ep. step: %d; Global step: %d; Time: %.2d:%.2d' % (junct_id, self.rl_agents[junct_id].ep_step, self.rl_agents[junct_id].step, sim_time/3600, int(sim_time/60)%60), level=2)

		if self.rl_agents[junct_id].ep_step == 0: # in the first step just observe (do nothing)
			log('[CONTROL] First step. Doing nothing...', level=2)

			# store last inputs for computing differences to current ones
			env_inputs = self.env.get_data_for_controller(junct_id)
			self.last_env_inputs[junct_id] = deepcopy(env_inputs)

			# store last loss for computing differences to current ones
			self.env.last_losses[junct_id] = 0.0

			# update agent variables
			self.rl_agents[junct_id].ep_step += 1
			self.rl_agents[junct_id].step += 1

			return None, False

		# keep track of environment statistics (over the last 2 minutes)
		self.ep_env_stats['total_travel_time'].append(np.mean(list(self.env.total_travel_time)[-24:]))
		self.ep_env_stats['veh_waiting'].append(np.mean(list(self.env.veh_waiting)[-24:]))
		self.ep_env_stats['veh_count'].append(np.mean(list(self.env.veh_count)[-24:]))
		self.ep_env_stats['avg_travel_time'].append(np.mean(list(self.env.avg_travel_time)[-24:]))
		self.ep_env_stats['avg_delay'].append(np.mean(list(self.env.avg_delay)[-24:]))
		self.ep_env_stats['avg_stop_time'].append(np.mean(list(self.env.avg_stop_time)[-24:]))
		self.ep_env_stats['avg_queue_length'].append(np.mean(list(self.env.avg_stop_time)[-24:]))

		# determine reward of the previous action
		reward, loss, rolling_loss = self.determine_reward(junct_id)
		self.ep_rewards[junct_id].append(reward)
		self.ep_losses[junct_id].append(rolling_loss)

		# check if the episode should be terminated early due to too much loss
		done = False
		if loss > self.env.MAX_LOSS:
			log('[CONTROL] Episode ended due to too much loss (loss=%.1f)' % (loss,), level=1)
			done = True
			reward = -100 # fixed penalty for "death"

		# backward step (NN training)
		if self.rl_agents[junct_id].ep_step > 1: # on the first step there was no previous action to learn from
			#log('[CONTROL] Backward pass', level=3)
			try:
				metrics = self.rl_agents[junct_id].backward(reward, terminal=done)
				# metrics: ['loss', 'mean_absolute_error', 'mean_q']
				self.ep_nn_losses[junct_id].append(metrics[0])
				self.ep_maes[junct_id].append(metrics[1])
				self.ep_qs[junct_id].append(metrics[2])
				log('[CONTROL] Metrics:', metrics, level=2)
			except:
				log('[CONTROL] Error calling backward step! (probably not enough entries)', level=2)
		else:
			log('[CONTROL] Second step. Dont learn anything yet...', level=2)

		if done:
			return None, True # there is no point in continuing...

		#log('[CONTROL] ------------------------------------------------', level=2)
		
		action = self.determine_next_action(junct_id)

		#if junct_id == 913:
		#	act = False

		# determine next action based on current policy (forward step)
		if act:
			self.take_action(junct_id, action, timeSta, time,  acycle)
		else:
			log('[CONTROL] Act = NO!', level=2)
			action = 0 

		# update agent variables
		self.rl_agents[junct_id].ep_step += 1
		self.rl_agents[junct_id].step += 1

		return action, done


	def determine_next_action(self, junct_id):
		raise NotImplementedError


	def take_action(self, junct_id, action, timeSta, time,  acycle):
		raise NotImplementedError


	def determine_reward(self, junct_id):
		# junct_id will be 0 in COORDINATION_TYPE == JOINT
		raise NotImplementedError


	def show_stats(self):
		#log('[STATS] Episode Losses:', ['%.1f' % (x,) for x in self.ep_losses], level=2)
		#self.ep_rewards[0] = 0 # reward at first step is always zero, because there was no action before
		#log('[STATS] Episode Rewards:', ['%.1f' % (x,) for x in self.ep_rewards], level=2)
		log('[STATS] Env. Loss:', self.losses[-1], level=1)
		log('[STATS] Reward:', self.rewards[-1], level=2)
		log('[STATS] NNet Losses:', self.nn_losses[-1], level=1)
		log('[STATS] MAE:', self.maes[-1], level=2)
		log('[STATS] Mean Q:', self.qs[-1], level=1)


	def save_stats(self):
		ep_length = np.mean([len(self.ep_losses[junct_id]) for junct_id in self.env.JUNCTIONS])
		if self.env.test_mode:
			with open(self.env.output_folder+'rl_stats_test.tsv', 'a') as f:
				if self.env.curr_episode == self.env.TEST_INTERVAL:
					f.write("episode\tep_length\tenv_loss\treward\tnnet_loss\tmae\tmean_q\n")
				f.write("%d\t%.1f\t\t%.1f\t\t%.1f\t%.1f\t\t%.2f\t%.2f\n" % (self.env.curr_episode,ep_length,self.losses[-1],self.rewards[-1],self.nn_losses[-1],self.maes[-1],self.qs[-1]))

			with open(self.env.output_folder+'env_stats_test.json', 'a') as f:
				f.write("%d\t%s\t%s\t%s\n" % (self.env.curr_episode,json.dumps(self.ep_losses), json.dumps(self.ep_rewards), json.dumps(self.ep_env_stats)))

			with open(self.env.output_folder+'rl_trajectories_test.txt', 'a') as f:
				f.write(json.dumps([self.ep_losses, self.ep_rewards, self.ep_trajectory])+'\n')
		else:
			with open(self.env.output_folder+'rl_stats.tsv', 'a') as f:
				if self.env.curr_episode == 1:
					f.write("episode\tep_length\tenv_loss\treward\tnnet_loss\tmae\tmean_q\n")
				f.write("%d\t%.1f\t\t%.1f\t\t%.1f\t%.1f\t\t%.2f\t%.2f\n" % (self.env.curr_episode,ep_length,self.losses[-1],self.rewards[-1],self.nn_losses[-1],self.maes[-1],self.qs[-1]))

			with open(self.env.output_folder+'env_stats.json', 'a') as f:
				f.write("%d\t%s\t%s\t%s\n" % (self.env.curr_episode,json.dumps(self.ep_losses), json.dumps(self.ep_rewards), json.dumps(self.ep_env_stats)))

			with open(self.env.output_folder+'rl_trajectories.txt', 'a') as f:
				f.write(json.dumps([self.ep_losses, self.ep_rewards, self.ep_trajectory])+'\n')


	def save_controller(self):
		# save controller object
		fname = self.env.scenario_path+'/'+self.env.output_folder+self.controller_name+'.checkpoint'
		log('[IO] Saving RL controller to file: ' + fname, level=2)
		checkpoint = {
			'rewards': self.rewards,
			'losses': self.losses,
			'nn_losses': self.nn_losses,
			'maes': self.maes,
			'qs': self.qs,
		}
		with open(fname, 'w') as f:
			pickle.dump(checkpoint, f)

		for junct_id in self.env.JUNCTIONS if self.env.COORDINATION_TYPE == self.env.INDEP else [0]:
			# save RL agent object
			fname_agent = self.env.scenario_path+'/'+self.env.output_folder+self.controller_name+'_agent_'+str(junct_id)+'.checkpoint'
			agent = {
				'step': self.rl_agents[junct_id].step,
				'memory': self.rl_agents[junct_id].memory,
				#'policy_eps': self.rl_agent.policy.inner_policy.eps,
			}
			log('[IO] Saving RL agent to file: ' + fname_agent, level=2)
			with open(fname_agent, 'w') as f:
				pickle.dump(agent, f)

			# save NN weights of RL agent
			fname_weights = self.env.scenario_path+'/'+self.env.output_folder+self.controller_name+'_'+str(junct_id)+'.h5'
			log('[IO] Saving NN controller weights to file: ' + fname_weights, level=2)
			self.rl_agents[junct_id].save_weights(fname_weights, overwrite=True)


	def load_controller(self, config):
		# load latest checkpoint of RL controller
		fname = self.env.scenario_path+'/'+self.env.output_folder+self.controller_name+'.checkpoint'
		log('[IO] Loading RL controller from file: ' + fname, level=2)
		try:
			# load controller checkpoint object
			with open(fname, 'r') as f:
				checkpoint = pickle.load(f)
			self.rewards = checkpoint['rewards']
			self.losses = checkpoint['losses']
			self.nn_losses = checkpoint['nn_losses']
			self.maes = checkpoint['maes']
			self.qs = checkpoint['qs']
		except Exception as e:
			log('[IO] File does not exist. Creating new empty controller...', level=2)


	def load_agents(self):
		for junct_id in self.env.JUNCTIONS if self.env.COORDINATION_TYPE == self.env.INDEP else [0]:
			try:
				# load RL agent object
				fname_agent = self.env.scenario_path+'/'+self.env.output_folder+self.controller_name+'_agent_'+str(junct_id)+'.checkpoint'
				log('[IO] Loading RL agent from file: ' + fname_agent, level=2)
				with open(fname_agent, 'r') as f:
					agent = pickle.load(f)
					self.rl_agents[junct_id].memory = agent['memory']
					log("[IO] Loaded memory size:", self.rl_agents[junct_id].memory.nb_entries, level=1)
					#self.policy.eps = agent['policy_eps']
					#self.rl_agent.policy.eps = self.policy.eps
					self.rl_agents[junct_id].step = agent['step']

				# load NN weights of RL agent
				fname_weights = self.env.scenario_path+'/'+self.env.output_folder+self.controller_name+'_'+str(junct_id)+'.h5'
				log('[IO] Loading NN controller weights from file: ' + fname_weights, level=2)
				self.rl_agents[junct_id].load_weights(fname_weights)
			except Exception as e:
				log('[IO] File does not exist. Creating new empty agent...', level=2)


