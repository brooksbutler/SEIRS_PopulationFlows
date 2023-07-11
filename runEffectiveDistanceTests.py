import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import stats
import pandas as pd

import ProcessData
import EffectiveDistance
import SEIRS
import EpiModels

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams.update({'font.family':'Times New Roman'})

np.random.seed(0)

confirmed_path = "covid_confirmed_usafacts_8_30_2021.csv"
pop_path = "covid_county_population_usafacts_8_30_2021.csv"
flows_path = "processed_OD_files/"
geo_dist_path = "sf12010countydistance500miles.csv"

h = 1/1000
het = True
dr, de, ds = 4, 3, 12
delay = 20
plot_pred_window = True
sample = 86
tau = 20

######################## SIMULATED SYSTEM ######################
Ns = ((pd.read_csv('Ns.csv', sep=',', header=None)).values).flatten()
Ws0 = (pd.read_csv('Ws0.csv', sep=',', header=None)).values
gammas0 = ((pd.read_csv('gammas0.csv', sep=',', header=None)).values).flatten()

n = len(Ws0)

numsteps = 100
Ws = []
gammas = []
for _ in range(numsteps):
    Ws.append(Ws0)
    gammas.append(gammas0)

# Ns, Fs, Ws, gammas = ProcessData.simulateFlows(n, numsteps, sparsity=0.2)

x0 = np.zeros((n,1))
source = 61
x0[source] = 0.2
s0 = 1-x0
e0 = np.zeros(x0.shape)
r0 = np.zeros(x0.shape)

# System parameters
beta, sigma, delta, alpha = (1e-2*np.random.uniform(0,10,n), 
                            1e-2*np.random.uniform(0,10,n), 
                            1.5e-3*np.random.uniform(0,10,n), 
                            1e-4*np.random.uniform(0,10,n))


SIRS = EpiModels.SIRS(beta, delta, alpha,Ns)
SIS = EpiModels.SIS(beta, alpha,Ns)

# Simulate model
h = 1
ss, es, xs, rs = SEIRS.simulate(h,s0,e0,x0,r0,Ws,beta,sigma,delta,alpha,gammas,Ns,numsteps)
# ss, xs, rs = SIRS.simulate(h,s0,x0,r0,Ws,gammas,numsteps)
# ss, xs = SIS.simulate(h,s0,x0,Ws,gammas,numsteps)

threshold = 0.01
arrived = np.zeros(n)
arrival_day = np.zeros(n)
for day, x in enumerate(xs):
    for i, x_i in enumerate(x):
        if (not arrived[i]) and (x_i > threshold):
            arrived[i] = True
            arrival_day[i] = day

D = EffectiveDistance.log_dist(Ws[10])

eff_distances = EffectiveDistance.compute_eff_dist(source, D)

node_loc = np.argsort(arrival_day)

slope_eff, intercept_eff, r_value_eff, p_value_eff, std_err_eff = stats.linregress(eff_distances[node_loc[:sample]],arrival_day[node_loc[:sample]])
print('------------------ SIMULATED DATA RESULTS ---------------------')
print("Effective Distance; R value: {}, P value: {}, std err: {}".format(r_value_eff, p_value_eff, std_err_eff))

x_eff = np.linspace(min(eff_distances), max(eff_distances), 100)
y_eff = EffectiveDistance.line(x_eff, slope_eff, intercept_eff)

errors, xs, ys, x_preds, y_preds = EffectiveDistance.timeVarPedict(tau,arrival_day,node_loc,Ws,Ns,figname="t_var_predict_sim",plot_pred_window=plot_pred_window)

y_eff_pred = slope_eff*eff_distances + intercept_eff*np.ones(eff_distances.shape)
error_eff = np.sqrt(np.mean((y_eff_pred - arrival_day)**2))

print("RMS total error: {}".format(error_eff))
print("RMS error for k-day prediction windows")
for k in range(1,15):
    e, e_k = EffectiveDistance.next_k_error(k, errors)
    print(k,e)
    # for ee in e_k:
    #     print("\t", ee)

plt.figure(figsize=(4,3))
plt.plot(eff_distances,arrival_day,  'b.')
plt.plot(x_eff, y_eff, 'k--')
plt.grid()
plt.ylabel('Arrival time')
plt.xlabel("$D_{eff}$")
plt.tight_layout()
plt.savefig("figures/sim_MN_arrival_vs_effdist.png", dpi=300)

plt.show()
