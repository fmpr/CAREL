#!/bin/bash

AIMSUN_PATH="/home/rodr/Aimsun_Next_8_2_2/"
CODE_PATH="/home/rodr/code/deep-rl-traffic-lights/"
SCENARIO_PATH="${CODE_PATH}scenarios/Single4WayIntersec_RealDemand_base/"
SCENARIO_FILE="${SCENARIO_PATH}scenario.ang"
CONFIG_FILE="${SCENARIO_PATH}scenario.conf"
TARGET=931
RUN=$2

if [ "$RUN" = "" ]; then
        RUN=1
fi

printf '[INPUT]\nSCENARIO_FILE = '$SCENARIO_FILE'\nCONFIG_FILE = '$CONFIG_FILE'\nTARGET = '$TARGET'\nRUN = '$RUN'\n' > args.in

if [ "$1" != "continue" ]; then
	rm rl.log
	rm -R runs/$RUN
	#rm runs/$RUN/rl_performance.tsv
	#rm runs/$RUN/rl_stats.tsv
	#rm runs/$RUN/rl_trajectories.txt
	#rm runs/$RUN/env_stats.json
	#rm runs/$RUN/rl_performance_test.tsv
        #rm runs/$RUN/rl_stats_test.tsv
        #rm runs/$RUN/rl_trajectories_test.txt
        #rm runs/$RUN/env_stats_test.json
	#rm runs/$RUN/env.checkpoint
	#rm runs/$RUN/controller.checkpoint
	#rm runs/$RUN/controller_*.h5
	#rm runs/$RUN/controller_agent_*.checkpoint
fi

mkdir -p runs/$RUN

case $1 in
	debug)
		${AIMSUN_PATH}aconsole -project $SCENARIO_FILE -cmd execute -target $TARGET -log_file ${SCENARIO_PATH}rl.log -v
		;;
	continue)
		${AIMSUN_PATH}aconsole -project $SCENARIO_FILE -cmd execute -target $TARGET -log_file ${SCENARIO_PATH}rl.log -v
		;;
	run)
		${AIMSUN_PATH}aconsole -project $SCENARIO_FILE -cmd execute -target $TARGET -log_file ${SCENARIO_PATH}rl.log -v | egrep "\[DeepRL\]\[L1\]"
		;;
	verbose)
		${AIMSUN_PATH}aconsole -project $SCENARIO_FILE -cmd execute -target $TARGET -log_file ${SCENARIO_PATH}rl.log -v | egrep "\[DeepRL\]\[L[12]\]"
		;;
	*)
		printf 'INVALID OPTION: '$1'\nVALID OPTIONS ARE: run, verbose, continue, debug\n'
		;;
esac

