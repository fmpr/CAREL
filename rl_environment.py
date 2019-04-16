from AAPI import *
from PyANGKernel import *
from utils import *
from collections import deque
from rl_controller import RLController
import os 
import shutil
import numpy as np
import pickle, json
import configparser

class RLEnvironment(object):
	'''
	Environment class
	'''
	def __init__(self, scenario_path, config_file, env_name='env'):
		self.scenario_path = scenario_path
		self.env_name = env_name

		# predefined constants for control types
		self.PHASE_SELECTION = 1
		self.TIME_EXTENSION = 2
		self.FIXED_TIMES = 3

		# predefined constants for coordination types
		self.INDEP = 1
		self.JOINT = 2

		# status variables
		self.curr_episode = 1 # keeps track of the current episode 
		self.curr_phases = None # keeps track of the current phase
		self.curr_tl_cycles = None # keeps track of the current cycle
		self.curr_timings = None # keeps track of the current signal timings (if applicable)
		self.times_since_green = None # keeps track of the time since the last green for each intersection
		self.last_losses = None

		# load input arguments for environment
		self.args = configparser.ConfigParser()
		self.args.read(config_file)
		self.run = int(self.args["INPUT"]["RUN"])
		self.output_folder = "runs/%d/" % (self.run,)
		log('[INIT] Output folder:', self.output_folder, level=1)

		#shutil.rmtree(self.output_folder, ignore_errors=True)
		#os.makedirs(self.output_folder)

		# configuration options
		self.config = None

		# aimsun model object
		self.aimsun_model = None

		# RL controller object
		self.rl_controller = None


	def initialize(self):
		# set random seed for replication
		log('[INIT] Setting random seed for replication:', self.RANDOM_SEED, level=2)
		self.aimsun_model = GKSystem.getSystem().getActiveModel()
		replication = self.aimsun_model.getCatalog().find(self.REPLICATION_ID)
		replication.setRandomSeed(self.RANDOM_SEED)

		# load OD demand data from file
		if self.DEMAND_FILE != 'None':
			self.load_demand_ODs()

		# initialize status variables
		self.curr_phases = {junct_id:0 for junct_id in self.JUNCTIONS} # keeps track of the current phase
		self.curr_tl_cycles = {junct_id:0 for junct_id in self.JUNCTIONS} # keeps track of the current cycle
		self.last_losses = {junct_id:0.0 for junct_id in self.JUNCTIONS} if self.COORDINATION_TYPE == self.INDEP else {0:0.0}

		# define initial signal timings
		if self.CONTROL_TYPE == self.TIME_EXTENSION or self.CONTROL_TYPE == self.FIXED_TIMES:
			self.curr_timings = {junct_id:np.array(self.INIT_TIMINGS[junct_id]) for junct_id in self.JUNCTIONS}
			if self.COORDINATION_TYPE == self.JOINT or self.CONTROL_TYPE == self.FIXED_TIMES:
				self.MAIN_JUNCTION = self.JUNCTIONS.keys()[0] # must define a main/central junction for deciding when to act
		elif self.CONTROL_TYPE == self.PHASE_SELECTION:
			self.curr_timings = {junct_id:60*np.ones(len(self.JUNCTIONS[junct_id])) for junct_id in self.JUNCTIONS}
			self.times_since_green = {junct_id:np.zeros(len(self.JUNCTIONS[junct_id])) for junct_id in self.JUNCTIONS}
		
		# apply initial timings
		for junct_id in self.JUNCTIONS:
			self.set_phase_timings(junct_id, self.curr_timings[junct_id])


		# check if this is a test simulatation run - i.e. no learning and no exploration
		self.test_mode = self.curr_episode % self.TEST_INTERVAL == 0
		log("[CONTROL] Test mode:", self.test_mode, level=1)

		# global variables for collecting statistics
		self.stats_count = {}
		self.total_travel_time = deque(maxlen=self.STATS_QUEUE_LEN)
		self.veh_waiting = deque(maxlen=self.STATS_QUEUE_LEN)
		self.veh_count = deque(maxlen=self.STATS_QUEUE_LEN)
		self.avg_travel_time = deque(maxlen=self.STATS_QUEUE_LEN)
		self.avg_delay = deque(maxlen=self.STATS_QUEUE_LEN)
		self.avg_stop_time = deque(maxlen=self.STATS_QUEUE_LEN) 
		self.avg_queue_length = deque(maxlen=self.STATS_QUEUE_LEN) 
		self.tl_counts = {}
		self.tl_densities= {}
		self.tl_occupancies = {}
		self.tl_speeds = {}
		self.tl_node_delays = {}
		self.tl_delays = {}
		self.tl_delays_old = {}
		self.tl_queue_lenghts = {}
		self.tl_queue_lenghts_old = {}
		for junct_id in self.JUNCTIONS:
			self.stats_count[junct_id] = 0
			self.tl_counts[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
			self.tl_densities[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
			self.tl_occupancies[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
			self.tl_speeds[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
			self.tl_node_delays[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN)
			self.tl_delays[junct_id] = {sec_id:deque(maxlen=self.STATS_QUEUE_LEN) for sec_id in self.JUNCTIONS[junct_id]}
			self.tl_delays_old[junct_id] = {sec_id:0.0 for sec_id in self.JUNCTIONS[junct_id]}
			self.tl_queue_lenghts[junct_id] = {sec_id:[deque(maxlen=self.STATS_QUEUE_LEN) for lane in range(self.JUNCTIONS[junct_id][sec_id])] for sec_id in self.JUNCTIONS[junct_id]}
			self.tl_queue_lenghts_old[junct_id] = {sec_id:[0.0 for lane in range(self.JUNCTIONS[junct_id][sec_id])] for sec_id in self.JUNCTIONS[junct_id]}
		

	def initialize_controller(self):
		# create new RL controller object
		log('[INIT] Creating new RL controller object...', level=1)
		self.rl_controller = RLController(self)
		self.rl_controller.load_controller(self.config)
		self.rl_controller.initialize()
		raise NotImplementedError


	def step(self, time, timeSta, timeTrans, acycle):
		if time <= self.WARMUP_DURATION:
			return

		# determine the simulation step number
		sim_time = time - timeTrans # do not count warmup
		#step = time / acycle

		if sim_time < 1:
			# on the first step, set initial phase timings
			log('[ENV] First simulation step! Setting initial phase timings... ', level=1)
			for junct_id in self.JUNCTIONS:
				self.set_phase_timings(junct_id, self.curr_timings[junct_id])

		# check if episode has reached the end
		if sim_time >= self.EPISODE_DURATION:
			log('[ENV] Episode reached maximum duration', level=1)
			self.start_new_episode()
			return

		# check if new statistics are available for collection - DATA COLLECTION
		if AKIEstIsNewStatisticsAvailable():
			# update measurements collected so far
			self.update_stats(sim_time, timeSta)

		# get current phases for each junction
		phases = {junct_id:ECIGetCurrentPhase(junct_id) for junct_id in self.JUNCTIONS}

		if self.CONTROL_TYPE == self.PHASE_SELECTION:
			self.curr_phases = phases

			# increment times since last green for all phases
			for junct_id in self.JUNCTIONS:
				self.times_since_green[junct_id] += acycle

			# check if enough time has passed since the last action - TIME FOR ACTION!
			if not sim_time % self.CONTROL_INTERVAL_SEC:
				if self.COORDINATION_TYPE == self.JOINT:
					actions, done = self.rl_controller.control(sim_time, timeSta, time,  acycle, 0)
					if done:
						self.start_new_episode()
						return

					# reset times since last green for the phase that just got the green 
					for junct_id in JUNCTIONS:
						self.times_since_green[junct_id][actions[junct_id]] = 0.0 
				
				elif self.COORDINATION_TYPE == self.INDEP:
					for junct_id in self.JUNCTIONS:
						action, done = self.rl_controller.control(sim_time, timeSta, time,  acycle, junct_id)
						if done:
							self.start_new_episode()
							return

						# reset times since last green for the phase that just got the green 
						self.times_since_green[junct_id][action] = 0.0 

		elif self.CONTROL_TYPE == self.TIME_EXTENSION or self.CONTROL_TYPE == self.FIXED_TIMES:
			
			if self.COORDINATION_TYPE == self.JOINT or self.CONTROL_TYPE == self.FIXED_TIMES:
				# check if a new cycle as started - TIME FOR ACTION!
				if phases[self.MAIN_JUNCTION] != self.curr_phases[self.MAIN_JUNCTION]: 
					self.curr_phases[self.MAIN_JUNCTION] = phases[self.MAIN_JUNCTION]
					if self.curr_phases[self.MAIN_JUNCTION] == 1: # just finished a complete cycle
						self.curr_tl_cycles[self.MAIN_JUNCTION] += 1
						if not self.curr_tl_cycles[self.MAIN_JUNCTION] % self.CONTROL_INTERVAL_TL: # only take actions every N cycles
							if self.CONTROL_TYPE == self.TIME_EXTENSION:
								actions, done = self.rl_controller.control(sim_time, timeSta, time,  acycle, 0)
							elif self.CONTROL_TYPE == self.FIXED_TIMES:
								actions, done = self.rl_controller.control(sim_time, timeSta, time,  acycle, self.MAIN_JUNCTION, act=False)

							if done:
								self.start_new_episode()
								return

							# reset statistics collected so far (for all junctions)
							for junct_id in self.JUNCTIONS:
								self.reset_stats(junct_id)

			elif self.COORDINATION_TYPE == self.INDEP:
				for junct_id in self.JUNCTIONS:
					# check if a new cycle as started - TIME FOR ACTION!
					if phases[junct_id] != self.curr_phases[junct_id]:
						self.curr_phases[junct_id] = phases[junct_id]
						if self.curr_phases[junct_id] == 1: # just finished a complete cycle
							self.curr_tl_cycles[junct_id] += 1
							if not self.curr_tl_cycles[junct_id] % self.CONTROL_INTERVAL_TL: # only take actions every N cycles
								action, done = self.rl_controller.control(sim_time, timeSta, time,  acycle, junct_id)

								if done:
									self.start_new_episode()
									return

								# reset statistics collected so far
								self.reset_stats(junct_id)


	def set_phase_timings(self, junct_id, timings):
		log('[ENV] Setting timings for junction %d: %s' % (junct_id, str(timings)), level=2)
		for i in range(len(timings)):
			ECIChangeTimingPhase(junct_id, i*2+1, timings[i], -1)


	def update_stats(self, time, timeSta):
		# get system-wide statistics (e.g. for rewards)
		estad = AKIEstGetParcialStatisticsSystem(time, 0)
		if estad.report == 0:
			#log('[ENV] Count: %d; Total travel time: %.2f; Veh. waiting: %d' % (estad.count,estad.TTa,estad.vehsWaiting), level=3)
			if estad.count != 0:
				self.veh_count.append(estad.count)
				self.total_travel_time.append(estad.TotalTravelTime*3600)
				self.veh_waiting.append(estad.vehsWaiting)
				self.avg_travel_time.append(estad.TTa)
				self.avg_delay.append(estad.DTa)
				self.avg_stop_time.append(estad.STa)
				self.avg_queue_length.append(estad.virtualQueueAvg)
		else:
			log('[ERROR] Invalid system statistics report', level=1)

		debug = False
		if debug: log('[STATS] UPDATING STATS (%d,%d)' % (time,timeSta), level=1)

		nodeDelay = AKIEstGetPartialStatisticsNodeApproachDelay(913)
		if debug: log('[STATS] NODE DELAY: %f' % (nodeDelay,), level=1)

		# get junction status statistics
		for junct_id in self.JUNCTIONS:
			self.stats_count[junct_id] += 1

			junct_delay = AKIEstGetPartialStatisticsNodeApproachDelay(junct_id)
			self.tl_node_delays[junct_id].append(junct_delay)
			#if debug: log('[STATS] Node delay:', junct_id, junct_delay, level=1)

			# get detectors statistics
			for detector_id in self.DETECTORS[junct_id]:
				count = AKIDetGetCounterAggregatedbyId(detector_id, 0)
				if count >= 0:
					self.tl_counts[junct_id][detector_id].append(count)
				else:
					self.tl_counts[junct_id][detector_id].append(self.tl_counts[junct_id][detector_id][-1])
				
				density = AKIDetGetDensityAggregatedbyId(detector_id, 0)
				if density >= 0:
					self.tl_densities[junct_id][detector_id].append(density)
				else:
					self.tl_densities[junct_id][detector_id].append(self.tl_densities[junct_id][detector_id][-1])
				
				occupacy = AKIDetGetTimeOccupedAggregatedbyId(detector_id, 0)
				if occupacy >= 0:
					self.tl_occupancies[junct_id][detector_id].append(occupacy)
				else:
					self.tl_occupancies[junct_id][detector_id].append(self.tl_occupancies[junct_id][detector_id][-1])
				
				speed = AKIDetGetSpeedAggregatedbyId(detector_id, 0)
				if speed >= 0:
					self.tl_speeds[junct_id][detector_id].append(speed)
				else:
					self.tl_speeds[junct_id][detector_id].append(self.tl_speeds[junct_id][detector_id][-1])

				#if debug: log('[STATS] Count/Density/Occupacy:', junct_id, detector_id, count, density, occupacy, level=1)

			# get lanes statistics - queue lengths (inputs for deciding actions)
			for sec_id in self.JUNCTIONS[junct_id]:
				#for sec_id2 in self.JUNCTIONS[junct_id]:
				#	estad = AKIEstGetParcialStatisticsTurning(sec_id, sec_id2, time, 0)
				#	if debug: log("[STATS] DTa:", sec_id, sec_id2, estad.DTa)
				#	if debug: log("[STATS] STa:", sec_id, sec_id2, estad.STa)

				estad = AKIEstGetParcialStatisticsSection(sec_id, time, 0)
				if estad.report == 0:
					if debug: log("[STATS] Section DTa:", sec_id, estad.DTa)
					if debug: log("[STATS] Section STa:", sec_id, estad.STa)
					if estad.DTa >= 0:
						delay = estad.DTa
					else:
						delay = self.tl_delays_old[junct_id][sec_id]
					self.tl_delays[junct_id][sec_id].append(delay)
					self.tl_delays_old[junct_id][sec_id] = delay
					if debug: log('[STATS] Section Delay:', sec_id, estad.DTa, delay, level=1)
				else:
					log('[ERROR] Invalid section statistics report', level=1)

				for lane_no in range(self.JUNCTIONS[junct_id][sec_id]):
					estad = AKIEstGetParcialStatisticsSectionLane(sec_id, lane_no, time, 0)
					#estad = AKIEstGetCurrentStatisticsSectionLane(sec_id, lane_ix, 0)
					if estad.report == 0:
						# the following step is necessary due to a bug in Aimsun: LongQueueAvg is never reset and always incremented
						#if self.stats_count > 1: # because when stats_count = 1, tl_queue_lenghts_old is empty 
						queue_len = estad.LongQueueAvg - self.tl_queue_lenghts_old[junct_id][sec_id][lane_no]
						self.tl_queue_lenghts[junct_id][sec_id][lane_no].append(queue_len)
						if debug: log('[STATS] Queue length:', sec_id, lane_no, queue_len, level=1)
						
						self.tl_queue_lenghts_old[junct_id][sec_id][lane_no] = estad.LongQueueAvg
					else:
						log('[ERROR] Invalid lane statistics report', level=1)


	def reset_stats(self, junct_id):
		log("[ENV] Resetting stats counter for junction %d..." % (junct_id,), level=3)
		self.stats_count[junct_id] = 0
		#self.total_travel_time[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN)
		#self.veh_waiting[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN)
		#self.veh_count[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN)
		#self.avg_travel_time[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN)
		#self.avg_delay[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN)
		#self.avg_stop_time[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN) 
		#self.avg_queue_length[junct_id] = deque(maxlen=self.STATS_QUEUE_LEN) 
		#self.tl_delays[junct_id] = {x:[deque(maxlen=self.STATS_QUEUE_LEN) for i in range(self.JUNCTIONS[junct_id][x])] for x in self.JUNCTIONS[junct_id]}
		#self.tl_queue_lenghts[junct_id] = {x:[deque(maxlen=self.STATS_QUEUE_LEN) for i in range(self.JUNCTIONS[junct_id][x])] for x in self.JUNCTIONS[junct_id]}
		#self.tl_counts[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
		#self.tl_counts[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
		#self.tl_densities[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
		#self.tl_occupancies[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}
		#self.tl_speeds[junct_id] = {x:deque(maxlen=self.STATS_QUEUE_LEN) for x in self.DETECTORS[junct_id]}


	def eval_and_track_performance(self):
		# get global system-wide statistics (for the whole simulation! - for performance evaluation and comparison)
		estad = AKIEstGetGlobalStatisticsSystem(0)
		if estad.report == 0:
			log('[STATS] ------------------------ Global episode performance metrics:', level=2)
			log('[STATS] Average Travel Time: %.2f' % (estad.TTa,), level=2)
			log('[STATS] Average Delay: %.2f' % (estad.DTa,), level=2)
			log('[STATS] Average Stop Time: %.2f' % (estad.STa,), level=2)
			log('[STATS] Average Virtual Queue Length: %.2f' % (estad.virtualQueueAvg,), level=2)
			log('[STATS] Num. Vehicles Waiting To Enter: %.2f' % (estad.vehsWaiting,), level=2)

			if self.test_mode:
				with open(self.output_folder+'rl_performance_test.tsv', 'a') as f:
					if self.curr_episode == self.TEST_INTERVAL:
						f.write("Episode\tAvg Travel Time\tAvg Delay\tAvg Stop Time\tAvg Virt Queue\tVeh Waiting\n")
					f.write("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n" % (self.curr_episode,estad.TTa,estad.DTa,estad.STa,estad.virtualQueueAvg,estad.vehsWaiting))
			else:
				with open(self.output_folder+'rl_performance.tsv', 'a') as f:
					if self.curr_episode == 1:
						f.write("Episode\tAvg Travel Time\tAvg Delay\tAvg Stop Time\tAvg Virt Queue\tVeh Waiting\n")
					f.write("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n" % (self.curr_episode,estad.TTa,estad.DTa,estad.STa,estad.virtualQueueAvg,estad.vehsWaiting))
		else:
			log('[ERROR] Invalid global system statistics report', level=1)


	def get_data_for_controller(self, junct_id):
		# junct_id will be 0 in COORDINATION_TYPE == JOINT
		raise NotImplementedError


	def start_new_episode(self):
		# keep track of progress and show statistics so far
		if self.CONTROL_TYPE == self.FIXED_TIMES:
			junct_ids = [self.MAIN_JUNCTION]
		elif self.COORDINATION_TYPE == self.INDEP:
			junct_ids = self.JUNCTIONS
		elif self.COORDINATION_TYPE == self.JOINT:
			junct_ids = [0]
		else:
			raise Exception("Unexpected combination of CONTROL_TYPE and COORDINATION_TYPE")

		self.rl_controller.rewards.append(np.mean([np.mean(self.rl_controller.ep_rewards[junct_id]) for junct_id in junct_ids]))
		self.rl_controller.losses.append(np.mean([np.mean(self.rl_controller.ep_losses[junct_id]) for junct_id in junct_ids]))
		self.rl_controller.nn_losses.append(np.mean([np.mean(self.rl_controller.ep_nn_losses[junct_id]) for junct_id in junct_ids]))
		self.rl_controller.maes.append(np.mean([np.mean(self.rl_controller.ep_maes[junct_id]) for junct_id in junct_ids]))
		self.rl_controller.qs.append(np.mean([np.mean(self.rl_controller.ep_qs[junct_id]) for junct_id in junct_ids]))
		self.rl_controller.show_stats()
		self.rl_controller.save_stats()

		# evaluate global system performance during whole episode
		self.eval_and_track_performance()

		self.curr_episode += 1

		# save environment to disk
		self.save_env()

		# save controller to disk
		self.rl_controller.save_controller()

		if self.curr_episode > self.MAX_EPISODES:
			log('[ENV] Maximum number of episodes reached! Terminating...', level=1)
			ANGSetSimulationOrder(1, 0) # cancel simulation 
			#ANGSetSimulationOrder(3, 0) # stop simulation (does not work on cmd line interface)
		else:
			log('[ENV] Rewinding simulation...', level=1)
			log('[ENV] ---------------------------------------------------------------------------', level=1)
			log('[ENV] Episode number:', self.curr_episode, level=1)
			ANGSetSimulationOrder(2, 0) # rewind simulation


	def save_env(self):
		fname = self.scenario_path+'/'+self.output_folder+self.env_name+'.checkpoint'
		log('[IO] Saving RL environment checkpoint to file: ' + fname, level=2)

		checkpoint = {
			'RANDOM_SEED': self.RANDOM_SEED,
			'curr_episode': self.curr_episode,
		}

		with open(fname, 'w') as f:
			pickle.dump(checkpoint, f)


	def load_env(self, config_file):
		# load environment configuration from config file
		log('[IO] Loading config file: ' + config_file, level=2)
		self.config = configparser.ConfigParser()
		self.config.read(config_file)

		# global static variables
		self.REPLICATION_ID = int(self.config["GENERAL"]["REPLICATION_ID"])
		self.RANDOM_SEED = int(self.config["GENERAL"]["RANDOM_SEED"]) + self.run * 1000

		# environment config
		self.JUNCTIONS = eval(self.config["ENVIRONMENT"]["JUNCTIONS"])
		self.CONNECTIONS = eval(self.config["ENVIRONMENT"]["CONNECTIONS"])
		self.DETECTORS = eval(self.config["ENVIRONMENT"]["DETECTORS"])
		self.DEMAND_FILE = self.config["ENVIRONMENT"]["DEMAND_FILE"]

		self.WARMUP_DURATION = eval(self.config["ENVIRONMENT"]["WARMUP_DURATION"])
		self.EPISODE_DURATION = eval(self.config["ENVIRONMENT"]["EPISODE_DURATION"])
		self.MAX_LOSS = int(self.config["ENVIRONMENT"]["MAX_LOSS"])
		self.MAX_EPISODES = int(self.config["ENVIRONMENT"]["MAX_EPISODES"])
		self.TEST_INTERVAL = int(self.config["ENVIRONMENT"]["TEST_INTERVAL"])
		self.STATS_QUEUE_LEN = int(self.config["ENVIRONMENT"]["STATS_QUEUE_LEN"])

		# controller config
		if self.config["CONTROLLER"]["CONTROL_TYPE"] == "PHASE_SELECTION":
			self.CONTROL_TYPE = self.PHASE_SELECTION
		elif self.config["CONTROLLER"]["CONTROL_TYPE"] == "TIME_EXTENSION":
			self.CONTROL_TYPE = self.TIME_EXTENSION
		elif self.config["CONTROLLER"]["CONTROL_TYPE"] == "FIXED_TIMES":
			self.CONTROL_TYPE = self.FIXED_TIMES
		else:
			raise Exception("Invalid control type:", self.config["CONTROLLER"]["CONTROL_TYPE"])

		if self.config["CONTROLLER"]["COORDINATION_TYPE"] == "INDEPENDENT" or self.config["CONTROLLER"]["COORDINATION_TYPE"] == "INDEP": 
			self.COORDINATION_TYPE = self.INDEP
		elif self.config["CONTROLLER"]["COORDINATION_TYPE"] == "JOINT":
			self.COORDINATION_TYPE = self.JOINT
		else:
			raise Exception("Invalid coordination type:", self.config["CONTROLLER"]["COORDINATION_TYPE"])

		log('[INIT] Control type is %s (%d)' % (self.config["CONTROLLER"]["CONTROL_TYPE"], self.CONTROL_TYPE), level=1)
		log('[INIT] Coordination type is %s (%d)' % (self.config["CONTROLLER"]["COORDINATION_TYPE"], self.COORDINATION_TYPE), level=1)

		# the following are only relevant if CONTROL_TYPE = PHASE-SELECTION
		if self.CONTROL_TYPE == self.PHASE_SELECTION:
			self.CONTROL_INTERVAL_SEC = int(self.config["CONTROLLER"]["CONTROL_INTERVAL_SEC"])

		# the following are only relevant if CONTROL_TYPE = TIME-EXTENSION
		if self.CONTROL_TYPE == self.TIME_EXTENSION or self.CONTROL_TYPE == self.FIXED_TIMES:
			self.CONTROL_INTERVAL_TL = int(self.config["CONTROLLER"]["CONTROL_INTERVAL_TL"])
			self.INIT_TIMINGS = eval(self.config["CONTROLLER"]["INIT_TIMINGS"])

		fname = self.scenario_path+'/'+self.output_folder+self.env_name+'.checkpoint'
		log('[IO] Loading RL environment checkpoint from file: ' + fname, level=2)
		try:
			with open(fname, 'r') as f:
				checkpoint = pickle.load(f)

			self.RANDOM_SEED = checkpoint["RANDOM_SEED"] + 1
			self.curr_episode = checkpoint["curr_episode"]
		except:
			log('[IO] File does not exist. Creating new empty environment...', level=2)


	def load_demand_ODs(self):
		fname = self.scenario_path+'/'+self.config["ENVIRONMENT"]["DEMAND_FILE"]
		log('[IO] Loading demand data from file: ' + fname, level=2)
		with open(fname) as f:
			f.readline() # ignore first line
			i = 1
			for line in f:
				dt_str, mat = line.strip().split('\t')
				mat = json.loads(mat)

				# set OD matrix for corresponding timeslot
				matObj = self.aimsun_model.getCatalog().findObjectByExternalId("mat%d" % (i,))
				matObj.setTripsFromList(mat)

				i += 1
				if i > 48:
					break



