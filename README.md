# CAREL

CAREL is open-source callback-based framework for promoting the flexible evaluation of different deep RL configurations under a traffic simulation environment, as proposed in the paper:

> Rodrigues, F. and Azevedo, C. Towards Robust Deep Reinforcement Learning for Traffic Signal Control: Demand Surges, Incidents and Sensor Failures. In proc. of IEEE Intelligent Transportation Systems Conference (ITSC), 2019.

# Architecture

<img src="https://github.com/fmpr/CAREL/blob/master/framework.png" alt="Architecture" width="500"/>

CAREL is structured into two main objects: the Environment and the Controller. The first allows for the integration with the simulation environment being used through available APIs. It enables the transfer of simulation statistics for the overall experimental assessment and for the synchronous decision making between the controller and the simulation environment. The Controller handles the learning strategy, neural network architecture, reward function, action space and state space described above. In the current Python implementation, the Controller relies on Keras-RL library for all fundamental RL-related capabilities.

With the proposed modular design, different control algorithms and configurations can be easily implemented in CAREL and transferred to other simulation or real environments. In this first deployment, the simulation environment used was AIMSUN Next.

# Requirements

- Python 2.7 (required for compatibility with AIMSUN, which currently does not support Python 3)
- AIMSUN Next 8.2.2 (versions 8.X.X were not tested, but they should also be compatible)
- Tensorflow 1.12.0
- Keras 2.1.6
- Keras-RL

# Usage

## Running a scenario

The folder "scenarios" contains an example scenario. In order to run a scenario, you need to ensure that:
- The path definitions in the scripts "run_scenario.sh" and "aimsun_interface.py" (e.g. AIMSUN_PATH and CODE_PATH) are correct. Also remember to make sure that the system path (sys.path) is correctly defined in the beginning of the "aimsun_interface.py" file.
- In AIMSUN, the file "aimsun_interface.py" in the scenario folder is loaded under the tab "Aimsun Next API" of the properties of the Dynamic Scenario object. Ensure the path is correct. 

Once this is done, you can run a scenario simply by navigating to the scenario folder and running:

```shell
./run_scenario.sh debug
```

The file "scenario.conf" contains configuration variables of the scenario, such as type of controller, maximum number of episodes, episode duration, etc. 

## Analysing the results of a scenario

During execution, a scenario will log a lot of statistics and results to the folder "runs". For example, the files "rl_stats.tsv" and "rl_performance.tsv" contain statistics about the performance of the RL algorithm over the episodes of a single run. 

You can use the script "results_stats.py" to analyse and plot these results. For example:
```shell
python results_stats.py scenarions/Single4WayIntersec_RealDemand_base/runs
```

## Creating a new scenario

Creating a new scenario in CAREL is relatively easy. You can exploit the intuitive interface of AIMSUN to build a complex and realistic scenario, and then adapt the RL controller code accordingly. The recommended sequence of steps to create a new scenario is the following:

- Make a copy the demo scenario folder and use it a starting point
- Edit the "scenario.ang" in AIMSUN (e.g. if you want to make changes to the road network, include pedestrians pedestrians, etc.). Remember to ensure that the correct "aimsun_interface.py" file is loaded under the tab "Aimsun Next API" of the properties of the Dynamic Scenario object.
- Edit the "scenario.conf" according to reflect the changes made to the network
- Update the path definitions in the scripts "run_scenario.sh" and "aimsun_interface.py" (particularly, SCENARIO_PATH)

Once the new scenario is set up, you just need to adapt the code of the controller. I.e., edit the functions "determine_reward", "take_action", "determine_next_action", "build_rl_agents" and "get_data_for_controller" under the files "rl_controller_custom.py" and "rl_environment_custom.py" according to the new scenario definition.



