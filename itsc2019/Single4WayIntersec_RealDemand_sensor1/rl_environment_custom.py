from AAPI import *
from PyANGKernel import *
from utils import *
from rl_environment import RLEnvironment
from rl_controller_custom import RLControllerCustom
import numpy as np
import pickle, json
import configparser

class RLEnvironmentCustom(RLEnvironment):
	'''
	Customized RLEnvironment class for this particular scenario
	'''
	def __init__(self, scenario_path, config_file, env_name='env'):
		self.sensor_failure = [False for s in range(4)]
		super(RLEnvironmentCustom, self).__init__(scenario_path, config_file, env_name=env_name)


	def initialize_controller(self):
		# create new RL controller object
		self.rl_controller = RLControllerCustom(self)
		self.rl_controller.load_controller(self.config)
		self.rl_controller.initialize()


	def get_data_for_controller(self, junct_id):
		if self.CONTROL_TYPE == self.PHASE_SELECTION:
			window = [-1]
		elif self.CONTROL_TYPE == self.TIME_EXTENSION or self.CONTROL_TYPE == self.FIXED_TIMES:
			window = range(-self.stats_count[junct_id],0)
		else:
			raise Exception("Unknown CONTROL_TYPE: "+str(self.CONTROL_TYPE))

		env_inputs = np.zeros(4)
		env_inputs[0] = max(np.mean(np.array(self.tl_queue_lenghts[913][861][0])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][0])[window]))
		env_inputs[1] = max(np.mean(np.array(self.tl_queue_lenghts[913][861][1])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][1])[window]), 
							np.mean(np.array(self.tl_queue_lenghts[913][861][2])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][2])[window]), 
							np.mean(np.array(self.tl_queue_lenghts[913][861][3])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][3])[window]))
		env_inputs[2] = max(np.mean(np.array(self.tl_queue_lenghts[913][863][0])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][0])[window]))
		env_inputs[3] = max(np.mean(np.array(self.tl_queue_lenghts[913][863][1])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][1])[window]),
							np.mean(np.array(self.tl_queue_lenghts[913][863][2])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][2])[window]),
							np.mean(np.array(self.tl_queue_lenghts[913][863][3])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][3])[window]))

		
		# sensor failure with some probability
		for s in range(len(self.sensor_failure)):
			if self.sensor_failure[s]:
				if np.random.rand() < 0.05: # failure recover probability
					self.sensor_failure[s] = False
			else:
				if np.random.rand() < 0.01: # failure probability
					self.sensor_failure[s] = True

			if self.sensor_failure[s]:
				env_inputs[s] = 0.0
		
		print "Sensor failure:", self.sensor_failure

		window_size = len(np.array(self.tl_queue_lenghts[913][861][0])[window])
		log("[ENV] Inputs for RL: %s (window_size=%d)" % (', '.join(['%.1f' % (x,) for x in env_inputs]), window_size), level=3)
		return env_inputs


	def determine_reward(self, junct_id):

		if self.CONTROL_TYPE == self.PHASE_SELECTION:
			window = [-1]
		elif self.CONTROL_TYPE == self.TIME_EXTENSION or self.CONTROL_TYPE == self.FIXED_TIMES:
			window = range(-self.stats_count[junct_id],0)
		else:
			raise Exception("Unknown CONTROL_TYPE: "+str(self.CONTROL_TYPE))

		ttt = np.mean(list(self.total_travel_time)[-24:])
		n_waiting = np.mean(list(self.veh_waiting)[-24:])
		n_veh = np.mean(list(self.veh_count)[-24:])
		avg_tt = ttt / n_veh
		rolling_loss = (ttt + n_waiting*avg_tt) / (n_veh + n_waiting)

		# new version
		if junct_id == 913:
			q_lenths = [max(np.mean(np.array(self.tl_queue_lenghts[913][861][0])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][0])[window])),
						max(np.mean(np.array(self.tl_queue_lenghts[913][861][1])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][1])[window]), 
							np.mean(np.array(self.tl_queue_lenghts[913][861][2])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][2])[window]), 
							np.mean(np.array(self.tl_queue_lenghts[913][861][3])[window]), np.mean(np.array(self.tl_queue_lenghts[913][865][3])[window])),
						max(np.mean(np.array(self.tl_queue_lenghts[913][863][0])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][0])[window])),
						max(np.mean(np.array(self.tl_queue_lenghts[913][863][1])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][1])[window]),
							np.mean(np.array(self.tl_queue_lenghts[913][863][2])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][2])[window]),
							np.mean(np.array(self.tl_queue_lenghts[913][863][3])[window]), np.mean(np.array(self.tl_queue_lenghts[913][867][3])[window]))
						]
			total_queue_len = np.sum(np.array(q_lenths)**2)
		elif junct_id == 1047:
			q_lenths = [max(np.mean(np.array(self.tl_queue_lenghts[1047][866][0])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1037][0])[window])),
						max(np.mean(np.array(self.tl_queue_lenghts[1047][866][1])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1037][1])[window]),
							np.mean(np.array(self.tl_queue_lenghts[1047][866][2])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1037][2])[window]),
							np.mean(np.array(self.tl_queue_lenghts[1047][866][3])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1037][3])[window])),
						max(np.mean(np.array(self.tl_queue_lenghts[1047][1032][0])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1034][0])[window])),
						max(np.mean(np.array(self.tl_queue_lenghts[1047][1032][1])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1034][1])[window]),
							np.mean(np.array(self.tl_queue_lenghts[1047][1032][2])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1034][2])[window]),
							np.mean(np.array(self.tl_queue_lenghts[1047][1032][3])[window]), np.mean(np.array(self.tl_queue_lenghts[1047][1034][3])[window]))
						]
			total_queue_len = np.sum(np.array(q_lenths)**2)
		else:
			raise Exception("Invalid junct_id in rewards calculation")

		loss = total_queue_len + np.mean(np.array(self.veh_waiting)[-1])
		#loss = total_queue_len + np.mean(np.array(self.veh_waiting)[window])
		#loss = total_queue_len + np.mean(list(self.veh_waiting)[-1])
		#reward = -(loss - self.last_losses[junct_id]) #-self.veh_waiting[-1]
		reward = -1*(loss - self.last_losses[junct_id]) #-self.veh_waiting[-1]
		self.last_losses[junct_id] = loss

		
		log('[ENV] Rolling Loss: %.1f \tInstant Loss: %.1f \tReward: %.1f' % (rolling_loss, loss, reward), level=2)
		return reward, loss, rolling_loss


