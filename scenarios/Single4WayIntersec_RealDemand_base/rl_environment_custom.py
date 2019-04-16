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

		window_size = len(np.array(self.tl_queue_lenghts[913][861][0])[window])
		log("[ENV] Inputs for RL: %s (window_size=%d)" % (', '.join(['%.1f' % (x,) for x in env_inputs]), window_size), level=3)
		return env_inputs


