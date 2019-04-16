from AAPI import *

AIMSUN_PATH = '/home/rodr/Aimsun_Next_8_2_2'
CODE_PATH = '/home/rodr/code/deep-rl-traffic-lights'
SCENARIO_PATH = CODE_PATH+'/scenarios/Single4WayIntersec_RealDemand_lower'

# The following code is required in order to be able to import third-party packages
import sys
sys.path = ['', AIMSUN_PATH, SCENARIO_PATH, CODE_PATH, '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/home/rodr/.local/lib/python2.7/site-packages', '/usr/local/lib/python2.7/dist-packages', '/usr/local/lib/python2.7/dist-packages/html5lib-0.9999999-py2.7.egg', '/usr/local/lib/python2.7/dist-packages/bleach-1.5.0-py2.7.egg', '/usr/local/lib/python2.7/dist-packages/GPy-1.9.2-py2.7-linux-x86_64.egg', '/usr/local/lib/python2.7/dist-packages/paramz-0.9.1-py2.7.egg', '/usr/local/lib/python2.7/dist-packages/pystan-2.17.1.0-py2.7-linux-x86_64.egg', '/usr/local/lib/python2.7/dist-packages/Cython-0.28.3-py2.7-linux-x86_64.egg', '/usr/local/lib/python2.7/dist-packages/control-0.7.0-py2.7.egg', '/usr/local/lib/python2.7/dist-packages/operalib-0.2b2-py2.7.egg', '/usr/local/lib/python2.7/dist-packages/dill-0.2.8.2-py2.7.egg', '/usr/local/lib/python2.7/dist-packages/h5py-2.9.0-py2.7-linux-x86_64.egg', '/usr/lib/python2.7/dist-packages']

# Useful debug information
AKIPrintString('------------------------------------------------------------')
AKIPrintString('PATH Variables:')
AKIPrintString('Aimsun PATH: ' + AIMSUN_PATH)
AKIPrintString('Code PATH: ' + CODE_PATH)
AKIPrintString('Scenario PATH: ' + SCENARIO_PATH)
AKIPrintString('Python Executable: ' + str(sys.executable))
AKIPrintString('System PATH: ' + str(sys.path))
AKIPrintString('------------------------------------------------------------')

# Import required 3rd-party packages
from rl_environment_custom import RLEnvironmentCustom
from utils import *

# Global variables
env = None # RL environment

def AAPILoad():
	global SCENARIO_PATH
	global env

	# load environment
	log('[INIT] Loading environment...', level=1)
	env = RLEnvironmentCustom(SCENARIO_PATH, SCENARIO_PATH+'/args.in')
	env.load_env(env.args["INPUT"]["CONFIG_FILE"])

	# initialize environment
	log('[INIT] Initializing environment...', level=1)
	env.initialize()
	env.initialize_controller()

	return 0


def AAPIInit():
	return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
	return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
	global env

	# update environment
	env.step(time, timeSta, timeTrans, acycle)
	
	return 0


def AAPIFinish():
	AKIPrintString( 'AAPIFinish' )
	return 0


def AAPIUnLoad():
	AKIPrintString( 'AAPIUnLoad' )
	return 0

	
def AAPIPreRouteChoiceCalculation(time, timeSta):
	#AKIPrintString( 'AAPIPreRouteChoiceCalculation' )
	return 0

def AAPIEnterVehicle(idveh, idsection):
	return 0

def AAPIExitVehicle(idveh, idsection):
	return 0

def AAPIEnterPedestrian(idPedestrian, originCentroid):
	return 0

def AAPIExitPedestrian(idPedestrian, destinationCentroid):
	return 0

def AAPIEnterVehicleSection(idveh, idsection, atime):
	return 0

def AAPIExitVehicleSection(idveh, idsection, atime):
	return 0
