#!/bin/sh

for RUN in {1..10}
do
	./run_scenario.sh debug $RUN
done

