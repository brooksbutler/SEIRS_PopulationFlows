from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import stats

import pandas as pd
import datetime
import networkx as nx

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def log_dist(W):
    n, m = W.shape
    D = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if W[i,j] == 0:
                D[i,j] = np.inf
            else:
                D[i,j] = -np.log(W[i,j])
    return D

def w_tilde(i, infected_nodes, Ns, W):
    pop_sum = 0
    for j in infected_nodes:
        pop_sum += Ns[j]
    w_til = 0
    for j in infected_nodes:
        w_til += Ns[j]*W[i,j]
    return w_til/pop_sum

def D_tilde(W, Ns, infected_nodes):
    n, m = W.shape
    D_til = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if (i in infected_nodes):
                D_til[i,j] = 0
            elif ((i not in infected_nodes) and (j in infected_nodes)):
                w_t = w_tilde(i, infected_nodes, Ns, W)
                if w_t > 0:
                    D_til[i,j] = -np.log(w_t)
                else:
                    D_til[i,j] = np.inf
            elif ((j not in infected_nodes) and (j not in infected_nodes)):
                if W[i,j] > 0:
                    D_til[i,j] = -np.log(W[i,j])
                else:
                    D_til[i,j] = np.inf
    return D_til


def line(x,m,b):
    return m * x + b

def compute_eff_dist(source, D):
    n = len(D)
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_edge(i,j, weight=D[j,i])

    paths = nx.algorithms.shortest_path(G, source=source, weight='weight')

    eff_distances = np.zeros(n)
    for p in paths:
        path = paths[p]
        d = 0
        for i, edge in enumerate(path[1:]):
            d += D[edge,path[i]]
        eff_distances[path[-1]] = d

    return eff_distances

def getClosestDate(day, dates):
    deltas = []
    for d in dates:
        delta = day-d
        deltas.append(delta.days)
    d_arr = np.array(deltas)
    arg_min_date = np.argmin(np.abs(d_arr))
    return arg_min_date

def timeVarPedict(tau,dfs,node_loc,Ws,Ns,sim=True,dates=None,travel_data_dates=None,figname=None,plot_pred_window=False):
    errors, xs, ys, x_preds, y_preds = [], [], [], [], []

    for k in range(len(node_loc)-1):
        if k > tau:
            y = dfs[node_loc[k-tau:k+1]]
            
            if not sim:
                closest_week = getClosestDate(dates[node_loc[k]], travel_data_dates)
                W = Ws[closest_week]
            else:
                W = Ws[0]

            infected_tau = node_loc[:k-tau+1]
            D_til_tau = D_tilde(W,Ns,infected_tau)
            eff_dist_tau = compute_eff_dist(node_loc[0], D_til_tau)
            
            x = np.zeros(y.shape)
            for i, n in enumerate(node_loc[k-tau:k+1]):
                x[i] = eff_dist_tau[n]
            
            y[0] = 0
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            xs.append(x)
            ys.append(y)

            infected_current = node_loc[:k+1]
            D_til_cur = D_tilde(W,Ns,infected_current)
            eff_dist_cur = compute_eff_dist(node_loc[0], D_til_cur)

            x_pred = np.zeros(node_loc[k+1:].shape)
            for i, n in enumerate(node_loc[k+1:]):
                x_pred[i] = eff_dist_cur[n]
            

            y_pred = slope*x_pred + (intercept)*np.ones(x_pred.shape)
            y_true = dfs[node_loc[k+1:]]

            if np.min(y_pred) < np.max(y):
                shift = np.max(y) - np.min(y_pred) +1
                y_pred = slope*x_pred + (intercept+shift)*np.ones(x_pred.shape)


            x_preds.append(x_pred)
            y_preds.append(y_pred)
            
            errors.append(y_pred-y_true)
            
            if plot_pred_window and figname:
                if k % 5 == 0:
                    x_line = np.linspace(np.min(x_pred), np.max(x_pred), 100)
                    y_line = line(x_line, slope, intercept+shift)

                    plt.figure(figsize=(3,2.5))
                    plt.plot(x,y,"ko",label="Training",markersize=3)
                    plt.plot(x_pred,y_pred,"r^",label="Prediction",markersize=3)
                    plt.plot(x_line,y_line,"r--")
                    plt.plot(x_pred,y_true,"bs",label="True",markersize=3)
                    plt.xlabel("$D_{eff}$")
                    plt.ylabel("T")
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()

                    if figname:
                        plt.savefig("figures/"+figname+"_"+str(k), dpi=300)

    return errors, xs, ys, x_preds, y_preds

def next_k_error(k, errors):
    err = 0
    count = 0
    e_k = []
    for e in errors:
        if len(e) > k:
            e_vec = np.array(e)
            e_k.append(e_vec[:k])
            err += np.sqrt(np.mean(e_vec[:k]**2))
            count += 1

    return err/count, e_k