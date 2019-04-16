import sys, os
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib

scenario_path = sys.argv[1]
if scenario_path[-1] != "/":
	scenario_path += "/"
print "Scenario path:", scenario_path

paths = sys.argv[1:]
data_stats = {}
data_perf = {}
for path in os.listdir(scenario_path):
	if ("results_" not in path and "runs" not in path) or "10runs" in path:
		continue

	if path[-1] != "/":
		path += "/"

	path = scenario_path + path
	name = path[path[:-1].rindex("/"):].replace("/","")
	print "\n\nResutls name:", name

	data_stats[name] = []
	data_perf[name] = []
	for run in os.listdir(path):
		try:
			run = eval(run)
		except:
			print("skipping dir:", run)
			continue

		rl_stats_fname = "%s%d/rl_stats.tsv" % (path, run)
		rl_perf_fname = "%s%d/rl_performance.tsv" % (path, run)
		#rl_stats_fname_test = "%s%d/rl_stats_test.tsv" % (path, run)
		#rl_perf_fname_test = "%s%d/rl_performance_test.tsv" % (path, run)
		if "no event experience" in path or "no incident experience" in path or "no failure experience" in path:
			rl_stats_fname = "%s%d/rl_stats_test.tsv" % (path, run)
			rl_perf_fname = "%s%d/rl_performance_test.tsv" % (path, run)
		print "processing file:", rl_stats_fname
		#print "processing file:", rl_perf_fname
		f = open(rl_stats_fname)
		fp = open(rl_perf_fname)
		f.readline() # skip header
		fp.readline() # skip header
		max_len = 80
		if "Single" in path:
			#max_len = 160
			max_len = 80
			#if "runs" in path:
			#	max_len = 120
		elif "Double" in path:
			max_len = 80*4
		else:
			max_len = 80*2
		mat_stats = np.zeros((max_len,7))
		mat_perf = np.zeros((max_len,6))
		line_no = 0
		for line in f:
			#print line.strip()
			line = line.strip().replace("\t\t","\t")
			episode, ep_length, env_loss, reward, nnet_loss, mae, mean_q = line.split("\t")
			mat_stats[line_no,:] = np.array([float(episode), float(ep_length), float(env_loss), float(reward), float(nnet_loss), float(mae), float(mean_q)])

			line = fp.readline().strip().replace("\t\t","\t")
			if len(line) > 0:
				episode, avg_tt, avg_delay, avg_stop_time, avg_virt_queue, veh_waiting = line.split("\t")
				mat_perf[line_no,:] = np.array([float(episode), float(avg_tt), float(avg_delay), float(avg_stop_time), float(avg_virt_queue), float(veh_waiting)])

			line_no += 1
			if line_no >= max_len:
				break


		#if "experience" in path:
		if "no event experience" in path or "no incident experience" in path or "no failure experience" in path:
			mat_stats = mat_stats[20:,:]
			mat_perf = mat_perf[20:,:]

		data_stats[name].append(mat_stats)
		data_perf[name].append(mat_perf)
		f.close()
		fp.close()

		data_stats[name][-1] = np.array(data_stats[name][-1])
		data_perf[name][-1] = np.array(data_perf[name][-1])
		#print data_stats[name][-1].shape

	data_stats[name] = np.array(data_stats[name])
	data_perf[name] = np.array(data_perf[name])
	#print data_stats[name].shape
	#print data_perf[name].shape

	#print "debuG:", data_stats[name].shape # 10, 80, 7
	avgs_last1 = data_stats[name][:,-1:,:].mean(axis=1).mean(axis=0)
	avgs_all = data_stats[name][:,:,:].mean(axis=1).mean(axis=0)
	#medians_all = np.median(data_stats[name][:,:,:], axis=1).mean(axis=0)
	medians_all = np.median(data_stats[name][:,:,:], axis=0).mean(axis=0)
	print "\nAverage results for all episodes (over all runs):"
	print "Total runs:", len(data_stats[name])
	print "Total episodes:", avgs_last1[0]
	print "Episode length:", avgs_all[1]
	print "Env loss (mean):", avgs_all[2]
	print "Env loss (median):", medians_all[2]
	print "Reward (median):", medians_all[3]
	print "Mean Q (median):", medians_all[6]
	avgs_all = data_perf[name][:,:,:].mean(axis=1).mean(axis=0)
	#medians_all = np.median(data_perf[name][:,:,:], axis=1).mean(axis=0)
	medians_all = np.median(data_perf[name][:,:,:], axis=0).mean(axis=0)
	print "Avg Travel Time (median):", medians_all[1]
	print "Avg Delay (median):", medians_all[2]
	print "Avg Stop Time (median):", medians_all[3]
	print "Avg Virt Queue (median):", medians_all[4]
	print "Veh Waiting (median):", medians_all[5]
	print "%.2f & %.2f & %.2f" % (medians_all[1], medians_all[2], medians_all[3])

	avgs_last30 = data_stats[name][:,-30:,:].mean(axis=1).mean(axis=0)
	#medians_last30 = np.median(data_stats[name][:,-30:,:], axis=1).mean(axis=0)
	medians_last30 = np.median(data_stats[name][:,-30:,:], axis=0).mean(axis=0)
	print "\nAverage results for last 30 episodes (over all runs):"
	print "Episode length:", avgs_last30[1]
	print "Env loss (mean):", avgs_last30[2]
	print "Env loss (median):", medians_last30[2]
	print "Reward (median):", medians_last30[3]
	print "Mean Q (median):", medians_last30[6]
	avgs_last30 = data_perf[name][:,-30:,:].mean(axis=1).mean(axis=0)
	#medians_last30 = np.median(data_perf[name][:,-30:,:], axis=1).mean(axis=0)
	medians_last30 = np.median(data_perf[name][:,-30:,:], axis=0).mean(axis=0)
	print "Avg Travel Time (median):", medians_last30[1]
	print "Avg Delay (median):", medians_last30[2]
	print "Avg Stop Time (median):", medians_last30[3]
	print "Avg Virt Queue (median):", medians_last30[4]
	print "Veh Waiting (median):", medians_last30[5]
	print "%.2f & %.2f & %.2f" % (medians_last30[1], medians_last30[2], medians_last30[3])

	avgs_last10 = data_stats[name][:,-10:,:].mean(axis=1).mean(axis=0)
	#medians_last10 = np.median(data_stats[name][:,-10:,:], axis=1).mean(axis=0)
	medians_last10 = np.median(data_stats[name][:,-10:,:], axis=0).mean(axis=0)
	print "\nAverage results for last 10 episodes (over all runs):"
	print "Episode length:", avgs_last10[1]
	print "Env loss (mean):", avgs_last10[2]
	print "Env loss (median):", medians_last10[2]
	print "Reward (median):", medians_last10[3]
	print "Mean Q (median):", medians_last10[6]
	avgs_last10 = data_perf[name][:,-10:,:].mean(axis=1).mean(axis=0)
	#medians_last10 = np.median(data_perf[name][:,-10:,:], axis=1).mean(axis=0)
	medians_last10 = np.median(data_perf[name][:,-10:,:], axis=0).mean(axis=0)
	print "Avg Travel Time (median):", medians_last10[1]
	print "Avg Delay (median):", medians_last10[2]
	print "Avg Stop Time (median):", medians_last10[3]
	print "Avg Virt Queue (median):", medians_last10[4]
	print "Veh Waiting (median):", medians_last10[5]
	print "%.2f & %.2f & %.2f" % (medians_last10[1], medians_last10[2], medians_last10[3])

# Visualize the results
cix = 0
legend = []
for name in data_stats:
	ix = range(data_stats[name].shape[1])
	means = data_stats[name][:,:,2].mean(axis=0)
	stds = data_stats[name][:,:,2].std(axis=0)
	medians = np.median(data_stats[name][:,:,2], axis=0)
	perc10 = np.percentile(data_stats[name][:,:,2], 10, axis=0)
	perc20 = np.percentile(data_stats[name][:,:,2], 20, axis=0)
	perc80 = np.percentile(data_stats[name][:,:,2], 80, axis=0)
	perc90 = np.percentile(data_stats[name][:,:,2], 90, axis=0)

	cmap = matplotlib.cm.get_cmap('tab10')
	rgba = cmap(cix)
	#plt.plot(ix, means, 'r-', color=rgba)
	#plt.fill_between(ix, means - stds, means + stds, color=rgba, alpha=0.2)
	plt.plot(ix, medians, 'r-', color=rgba)
	#plt.plot(ix, 85*np.ones(len(ix)), 'r-', color=rgba)
	plt.fill_between(ix, perc10, perc90, color=rgba, alpha=0.2)
	#plt.fill_between(ix, perc20, perc80, color=rgba, alpha=0.2)
	#for i in range(10):
	#	plt.plot(data[name][i,:,2], 'rx')
	plt.ylim([0,300])
	plt.xlabel("episode")
	plt.ylabel("rolling loss")
	legend.append(name.replace("results_","").replace("_"," "))
	cix += 0.2

plt.legend(legend)
plt.show()


cix = 0
legend = []
print data_perf.keys()
#for name in ['results_phase_selection_(with failure experience)', 'results_phase_selection_(no failure experience)']:
for name in data_perf:
	ix = range(data_perf[name].shape[1])
	means = data_perf[name][:,:,1].mean(axis=0)
	stds = data_perf[name][:,:,1].std(axis=0)
	medians = np.median(data_perf[name][:,:,1], axis=0)
	perc10 = np.percentile(data_perf[name][:,:,1], 10, axis=0)
	perc20 = np.percentile(data_perf[name][:,:,1], 20, axis=0)
	perc80 = np.percentile(data_perf[name][:,:,1], 80, axis=0)
	perc90 = np.percentile(data_perf[name][:,:,1], 90, axis=0)

	cmap = matplotlib.cm.get_cmap('tab10')
	rgba = cmap(cix)
	#plt.plot(ix, means, 'r-', color=rgba)
	#plt.fill_between(ix, means - stds, means + stds, color=rgba, alpha=0.2)
	plt.plot(ix, medians, 'r-', color=rgba)
	#plt.plot(ix, 85*np.ones(len(ix)), 'r-', color=rgba)
	plt.fill_between(ix, perc10, perc90, color=rgba, alpha=0.2)
	#plt.fill_between(ix, perc20, perc80, color=rgba, alpha=0.2)
	#for i in range(10):
	#	plt.plot(data[name][i,:,2], 'rx')
	plt.ylim([0,1000])
	#plt.xlim([0,60])
	plt.xlabel("episode")
	plt.ylabel("total travel time")
	legend.append(name.replace("results_","").replace("_"," "))
	cix += 0.2

plt.legend(legend)
#plt.axvline(x=39, c='orange')
#plt.annotate('Scenario Transfer', xy=(40, 500), xytext=(50, 600),
#        arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=7.0),
#        )
#plt.legend(["Deep RL with transfer learning", "Deep RL on event scenario"])
plt.show()


plot_nnloss = False
if plot_nnloss:
	cix = 0
	legend = []
	for name in data_stats:
		ix = range(data_stats[name].shape[1])
		means = data_stats[name][:,:,4].mean(axis=0)
		stds = data_stats[name][:,:,4].std(axis=0)
		medians = np.median(data_stats[name][:,:,4], axis=0)
		perc10 = np.percentile(data_stats[name][:,:,4], 10, axis=0)
		perc20 = np.percentile(data_stats[name][:,:,4], 20, axis=0)
		perc80 = np.percentile(data_stats[name][:,:,4], 80, axis=0)
		perc90 = np.percentile(data_stats[name][:,:,4], 90, axis=0)

		cmap = matplotlib.cm.get_cmap('tab10')
		rgba = cmap(cix)
		#plt.plot(ix, means, 'r-', color=rgba)
		#plt.fill_between(ix, means - stds, means + stds, color=rgba, alpha=0.2)
		#plt.plot(ix, medians, 'r-', color=rgba)
		if name == "runs":
			#plt.plot(ix, data_stats[name][3,:,4], 'r-', color=rgba)
			plt.plot(ix, data_stats[name][0,:,4], 'r-', color=rgba)
		elif "phase" in name:
			plt.plot(ix, medians, 'r-', color=rgba)
		else:
			continue
		#plt.plot(ix, 85*np.ones(len(ix)), 'r-', color=rgba)
		#plt.fill_between(ix, perc10, perc90, color=rgba, alpha=0.2)
		#plt.fill_between(ix, perc20, perc80, color=rgba, alpha=0.2)
		#for i in range(10):
		#	plt.plot(data[name][i,:,2], 'rx')
		plt.ylim([0,14000])
		plt.xlabel("episode")
		plt.ylabel("nnet loss")
		legend.append(name.replace("results_","").replace("_"," "))
		cix += 0.2

	plt.axvline(x=39, c='orange')
	plt.annotate('Scenario Transfer', xy=(40, 5000), xytext=(50, 6000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=7.0),
            )
	plt.legend(["Deep RL with transfer learning", "Deep RL on event scenario"])
	plt.show()


