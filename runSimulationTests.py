import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.extras import isin
from numpy.random.mtrand import randint
from sklearn.metrics import mean_squared_error
from math import sqrt
import tqdm

import ProcessData
import SEIRS
import SEIRS2
import Plots

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def rms(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

np.random.seed(0)

# numsteps = 4000
numsteps = 400
n = 5 # number of nodes to simulate
h = 1 # step-size
het = True
# het = False
noise = True
# noise = False
noise_scale = 1e-2

print(noise_scale)
test_param_senstivity = False

# Initial conditions
e0 = np.random.uniform(0,1,(n,1))
s0 = 1-e0
x0 = np.zeros(e0.shape)
r0 = np.zeros(e0.shape)

print("s0 = {}".format(np.around(e0, decimals=3)))

# System parameters
if het:
    beta, sigma, delta, alpha = (1e-1*np.random.uniform(0,1,n), 
                                1e-1*np.random.uniform(0,1,n), 
                                1e-2*np.random.uniform(0,1,n), 
                                1e-2*np.random.uniform(0,1,n))
else:
    beta, sigma, delta, alpha = (.08*np.ones(n), 
                                .07*np.ones(n), 
                                .05*np.ones(n), 
                                .01*np.ones(n))

# Simulate flows data
Ns, Fs, Ws, gammas = ProcessData.simulateFlows(n, numsteps)

print('beta: {}'.format(np.around(beta, decimals=3)))
print('sigma: {}'.format(np.around(sigma, decimals=3)))
print('delta: {}'.format(np.around(delta, decimals=3)))
print('alpha: {}'.format(np.around(alpha, decimals=3)))
print('gamma: {}'.format(np.around(gammas[0], decimals=3)))
print(np.around(Ws[0], decimals=3))

# Simulate model
ss, es, xs, rs = SEIRS.simulate(h,s0,e0,x0,r0,Ws,beta,sigma,delta,alpha,gammas,Ns,numsteps)

# print(xs[-1])

if noise:
    for i in range(len(ss)):
        ss[i] = ss[i] + np.random.normal(loc=0, scale=noise_scale, size = ss[i].shape)
        es[i] = es[i] + np.random.normal(loc=0, scale=noise_scale, size = es[i].shape)
        xs[i] = xs[i] + np.random.normal(loc=0, scale=noise_scale, size = xs[i].shape)
        rs[i] = rs[i] + np.random.normal(loc=0, scale=noise_scale, size = rs[i].shape)

# print(ss[-1]+ es[-1]+ xs[-1]+ rs[-1])

beta_e, beta_x = .5*beta, beta
ss_2, es_2, xs_2, rs_2 = SEIRS2.simulate(h,s0,e0,x0,r0,Ws,beta_e,beta_x,sigma,delta,alpha,numsteps)

Q, Delta = SEIRS.buildID(ss,es,xs,rs,Ws,gammas,Ns,h, het)
Q2, Delta2 = SEIRS2.buildID(ss_2,es_2,xs_2,rs_2,Ws,h,het)

params = (np.linalg.pinv(Q)@Delta).flatten()
params2 = (np.linalg.pinv(Q2)@Delta2).flatten()

if het:
    beta_hat, sigma_hat, delta_hat, alpha_hat = params[::4], params[1::4], params[2::4], params[3::4]
    beta1_hat2, beta2_hat2, sigma_hat2, delta_hat2, alpha_hat2 = params2[::5], params2[1::5], params2[2::5], params2[3::5], params2[4::5]
else:
    beta_hat, sigma_hat, delta_hat, alpha_hat = params
    beta_hat, sigma_hat, delta_hat, alpha_hat = (beta_hat*np.ones(n), sigma_hat*np.ones(n), 
                                                 delta_hat*np.ones(n), alpha_hat*np.ones(n))
    beta1_hat2, beta2_hat2, sigma_hat2, delta_hat2, alpha_hat2 = params2
    beta1_hat2, beta2_hat2, sigma_hat2, delta_hat2, alpha_hat2 = (beta1_hat2*np.ones(n), beta2_hat2*np.ones(n), sigma_hat2*np.ones(n), 
                                                                  delta_hat2*np.ones(n), alpha_hat2*np.ones(n))


print('beta error: {0:.3f}'.format(rms(beta,beta_hat)))
print('sigma error: {0:.3f}'.format(rms(sigma,sigma_hat)))
print('delta error: {0:.3f}'.format(rms(delta,delta_hat)))
print('alpha error: {0:.3f}'.format(rms(alpha,alpha_hat)))

ss_p, es_p, xs_p, rs_p = SEIRS.simulate(h,s0,e0,x0,r0,Ws,beta_hat,sigma_hat,delta_hat,alpha_hat,gammas,Ns,numsteps)
ss_p2, es_p2, xs_p2, rs_p2 = SEIRS2.simulate(h,s0,e0,x0,r0,Ws,beta1_hat2,beta2_hat2,sigma_hat2,delta_hat2,alpha_hat2,numsteps)

# Plots.compareSEIRS(ss,es,xs,rs,ss_p,es_p,xs_p,rs_p, figname="Simulation_ParamFit_wNoise")
Plots.compareSEIRS_vert(ss,es,xs,rs,ss_p,es_p,xs_p,rs_p, figname="Simulation_ParamFit_wNoise_vert")

t = numsteps-1

if test_param_senstivity:
    # Parameter sensitivty tests
    lim = max([max(gammas[i]) for i in range(len(gammas))])
    print(lim)
    # params_list = [beta, sigma, delta, alpha, gammas]
    params_list = [beta, gammas]
    # d_param = np.linspace(-.001,.1, 100)
    d_param = np.linspace(0,1.5, 10)
    x_star = xs[t]
    dx_star = np.zeros((len(params_list),len(d_param)))

    for i, p in enumerate(params_list):
        print(i)
        loop = tqdm.tqdm(total=len(d_param),position=0,leave=False)
        for j, dp in enumerate(d_param):
            ps_cp = params_list.copy()
            if isinstance(ps_cp[i], list):
                ls = []
                for l in ps_cp[i]:
                    ls.append(l*dp)
                ps_cp[i] = ls  
            else:
                ps_cp[i] = p*dp
            ss_new, es_new, xs_new, rs_new = SEIRS.simulate(h,s0,e0,x0,r0,Ws,ps_cp[0],sigma, delta, alpha,ps_cp[1],Ns,numsteps)
            dx_star[i,j] = np.mean(xs_new[t]-x_star)
            loop.update(1)

    # legends = ['beta', 'sigma', 'delta', 'alpha', 'gammas']
    legends = ['beta', 'gammas']
    plt.figure()
    plt.plot(d_param,dx_star.T)
    plt.grid()
    # plt.ylim(-1,1)
    plt.legend(legends)
    plt.xlabel('Multiplier')
    # plt.xlabel("$\delta \psi$")
    plt.ylabel('$\delta x^*$')

    # plt.figure()
    # plt.plot(d_param,dx_star[1])
    # plt.grid()
    # plt.xlabel('Multiplier')
    # # plt.xlabel("$\delta \psi$")
    # plt.ylabel('$\delta x^*$')


# Plots.plotSEIRS(ss_new, es_new, xs_new, rs_new)

plt.show()

# print(" Done")