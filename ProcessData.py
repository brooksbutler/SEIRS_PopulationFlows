import numpy as np

def simulateFlows(n, numsteps, sparsity=None):
    Fs, Ws, gammas = [], [], []
    Ns = np.random.randint(10000, 200000, n)
    V1 = .1*np.random.randint(1, 100, (n,n))
    
    F = V1.T@V1
    for i in range(n):
            F[i,i] = 0

    if sparsity:
        for i in range(n):
            for j in range(n):
                if (i < j) and  (np.random.uniform(0,1) < sparsity):
                    F[i,j], F[j,i] = 0, 0

    for _ in range(numsteps):
        W = np.zeros((n,n))
        
        for i in range(n):
            W[:,i] = F[:,i]/np.sum(F[:,i]) 
            
        Fs.append(F)
        Ws.append(W)
        gammas.append(np.sum(F,axis=0)/Ns)
    return Ns, Fs, Ws, gammas