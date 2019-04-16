import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from AAPI import *
from utils import *
from rl_controller import RLController
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


class RLControllerCustom(RLController):
	"""
	Customized RLController class for this particular scenario
	"""
	def __init__(self, env, controller_name='controller'):
		super(RLControllerCustom, self).__init__(env, controller_name=controller_name)


	def build_nnet_phase_selection(self):
		model = Sequential()
		model.add(Flatten(input_shape=(1,4+4)))
		#model.add(BatchNormalization())
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		#model.add(Dense(16))
		#model.add(Activation('relu'))
		model.add(Dense(4, activation='linear',
                #kernel_regularizer=regularizers.l2(0.05),
                #activity_regularizer=regularizers.l1(0.01)
		))

		return model


	def build_nnet_time_extension(self):
		model = Sequential()
		model.add(Flatten(input_shape=(1,4+4)))
		#model.add(BatchNormalization())
		model.add(Dense(12))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(12))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		#model.add(Dense(16))
		#model.add(Activation('relu'))
		model.add(Dense(9, activation='linear',
                #kernel_regularizer=regularizers.l2(0.05),
                #activity_regularizer=regularizers.l1(0.01)
		))

		return model

	def build_rl_agents(self, test_mode):
		log('[INIT] Building RL agents...', level=2)
		self.models = {}
		self.rl_agents = {}

		for junct_id in self.env.JUNCTIONS if self.env.COORDINATION_TYPE == self.env.INDEP else [0]:
			if self.env.CONTROL_TYPE == self.env.PHASE_SELECTION:
				self.models[junct_id] = self.build_nnet_phase_selection()
				memory = SequentialMemory(limit=10000, window_length=1)
				if test_mode:
					policy = EpsGreedyQPolicy(eps=0.0)
				else:
					policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=.1, value_test=.05, nb_steps=1000)
				
				self.rl_agents[junct_id] = DQNAgent(model=self.models[junct_id], nb_actions=4, memory=memory, nb_steps_warmup=0,
				               enable_dueling_network=True, dueling_type='avg', target_model_update=0.005, policy=policy)
				self.rl_agents[junct_id].compile(Adam(lr=1e-4), metrics=['mae'])

			elif self.env.CONTROL_TYPE == self.env.TIME_EXTENSION or self.env.CONTROL_TYPE == self.env.FIXED_TIMES:
				self.models[junct_id] = self.build_nnet_time_extension()
				memory = SequentialMemory(limit=10000, window_length=1)
				if test_mode:
					policy = EpsGreedyQPolicy(eps=0.0)
				else:
					#policy = BoltzmannQPolicy()
					#policy = EpsGreedyQPolicy(eps=0.5)
					policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=.1, value_test=.05, nb_steps=1000)
				
				# enable the dueling network
				# you can specify the dueling_type to one of {'avg','max','naive'}
				self.rl_agents[junct_id] = DQNAgent(model=self.models[junct_id], nb_actions=9, memory=memory, nb_steps_warmup=0,
				               enable_dueling_network=True, dueling_type='avg', target_model_update=0.005, policy=policy)
				self.rl_agents[junct_id].compile(Adam(lr=1e-3), metrics=['mae'])

			else:
				raise Exception("Invalid CONTROL_TYPE: "+self.env.CONTROL_TYPE)


	def determine_next_action(self, junct_id):
		#log('[CONTROL] Determining next action based on current policy...', level=3)

		# get inputs from the environment
		env_inputs = self.env.get_data_for_controller(junct_id)

		# prepare inputs for the neural network
		if self.env.CONTROL_TYPE == self.env.PHASE_SELECTION:
			inputs = np.concatenate([env_inputs, self.env.times_since_green[junct_id]/10.0], axis=0)
			log("[ENV] Times since green: %s" % (', '.join(['%.1f' % (x,) for x in self.env.times_since_green[junct_id]/10.0])), level=3)
		elif self.env.CONTROL_TYPE == self.env.TIME_EXTENSION or self.env.CONTROL_TYPE == self.env.FIXED_TIMES:
			#inputs = env_inputs - self.last_env_input
			#inputs = np.concatenate([env_inputs - self.last_env_inputs[junct_id], self.env.curr_timings[junct_id]/60.0], axis=0)
			inputs = np.concatenate([env_inputs, self.env.curr_timings[junct_id]/60.0], axis=0)
			log("[ENV] Current timings: %s" % (', '.join(['%.1f' % (x,) for x in self.env.curr_timings[junct_id]/60.0])), level=3)

		# forward pass on the neural network to determine next action
		action = self.rl_agents[junct_id].forward(inputs)

		# store last inputs for computing differences to current ones
		self.last_env_inputs[junct_id] = deepcopy(env_inputs)

		return action


	def take_action(self, junct_id, action, timeSta, time,  acycle):
		log('[CONTROL] Performing action %d on junction %d' % (action,junct_id), level=3)
		if self.env.CONTROL_TYPE == self.env.PHASE_SELECTION:
			#VALID_PHASES = [2,4,6,8]
			VALID_PHASES = [1,3,5,7]

			#change = np.random.rand() < 0.2
			#if change:
			#	action = int(np.random.rand()*len(VALID_PHASES))
			#else:
			#	action = VALID_PHASES.index(ECIGetCurrentPhase(self.env.JUNCTION_ID))

			# determine what the new phase will be
			new_phase = VALID_PHASES[action]
			if new_phase != self.env.curr_phases[junct_id]:
				log("[ENV] Setting NEW phase (with transition):", new_phase, level=2)
				ECIChangeDirectPhaseWithInterphaseTransition(junct_id, new_phase, timeSta, time,  acycle)
			else:
				log("[ENV] Keeping CURRENT phase:", self.env.curr_phases[junct_id], level=2)
				#ECIChangeDirectPhaseWithInterphaseTransition(junct_id, self.env.curr_phases[junct_id], timeSta, time,  acycle)


			# keep track of episode trajectory
			phase_one_hot = np.zeros(len(VALID_PHASES)).tolist()
			phase_one_hot[action] = 1
			self.ep_trajectory[junct_id].append(phase_one_hot)

		elif self.env.CONTROL_TYPE == self.env.TIME_EXTENSION or self.env.CONTROL_TYPE == self.env.FIXED_TIMES:
			# determine new signal timings
			if action == 0:
				pass # do nothing
			elif action == 1:
				self.env.curr_timings[junct_id][0] -= 5
			elif action == 2:
				self.env.curr_timings[junct_id][0] += 5
			elif action == 3:
				self.env.curr_timings[junct_id][1] -= 5
			elif action == 4:
				self.env.curr_timings[junct_id][1] += 5
			elif action == 5:
				self.env.curr_timings[junct_id][2] -= 5
			elif action == 6:
				self.env.curr_timings[junct_id][2] += 5
			elif action == 7:
				self.env.curr_timings[junct_id][3] -= 5
			elif action == 8:
				self.env.curr_timings[junct_id][3] += 5

			# ensure that all the signal durations are within the valid range
			for i in range(len(self.env.curr_timings[junct_id])):
				if self.env.curr_timings[junct_id][i] < 5:
					self.env.curr_timings[junct_id][i] = 5
				elif self.env.curr_timings[junct_id][i] > 60:
					self.env.curr_timings[junct_id][i] = 60

			# set new timings
			self.env.set_phase_timings(junct_id, self.env.curr_timings[junct_id])

			# keep track of episode trajectory
			self.ep_trajectory[junct_id].append(deepcopy(self.env.curr_timings[junct_id]).tolist())


