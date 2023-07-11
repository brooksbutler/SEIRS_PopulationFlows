import numpy as np
import torch
import cvxpy as cp

# Model dynamics equations
def s_dot(s, x, r, beta, alpha, gammas, W, Ns, useTorch):
    X, N, G = np.diag(x.flatten()), np.diag(Ns), np.diag(gammas)
    if useTorch:
        s, r, W = torch.Tensor(s), torch.Tensor(r), torch.Tensor(W)
        X, N, G = torch.Tensor(X), torch.Tensor(N), torch.Tensor(G)
        B, A  = torch.diag(beta), torch.diag(alpha), 
        return -(B@X+G)@s + (torch.linalg.inv(N)@W@G@N)@s + A@r    
    else:
        B, A = np.diag(beta), np.diag(alpha)
        return -(B@X+G)@s + (np.linalg.inv(N)@W@G@N)@s + A@r

def e_dot(s, e, x, beta, sigma, gammas, W, Ns, useTorch):
    X, N, G = np.diag(x.flatten()), np.diag(Ns), np.diag(gammas)
    if useTorch:
        s, e, W = torch.Tensor(s), torch.Tensor(e), torch.Tensor(W)
        X, N, G = torch.Tensor(X), torch.Tensor(N), torch.Tensor(G)
        B, Sig,  = torch.diag(beta), torch.diag(sigma)
        return B@X@s -(Sig+G)@e + (torch.linalg.inv(N)@W@G@N)@e
    else:
        B, Sig,  = np.diag(beta), np.diag(sigma)
        return B@X@s -(Sig+G)@e + (np.linalg.inv(N)@W@G@N)@e

def x_dot(e, x, sigma, delta, gammas, W, Ns, useTorch):
    N, G = np.diag(Ns), np.diag(gammas)
    if useTorch:
        e, x, W = torch.Tensor(e), torch.Tensor(x), torch.Tensor(W)
        N, G = torch.Tensor(N), torch.Tensor(G)
        Sig, D = torch.diag(sigma), torch.diag(delta)
        return Sig@e -(D+G)@x + (torch.linalg.inv(N)@W@G@N)@x    
    else:
        Sig, D = np.diag(sigma), np.diag(delta)
        return Sig@e -(D+G)@x + (np.linalg.inv(N)@W@G@N)@x

def r_dot(x, r, delta, alphas, gammas, W, Ns, useTorch):
    N, G = np.diag(Ns), np.diag(gammas)
    if useTorch:
        x, r, W = torch.Tensor(x), torch.Tensor(r), torch.Tensor(W)
        N, G = torch.Tensor(N), torch.Tensor(G)
        D, A = torch.diag(delta), torch.diag(alphas)
        return D@x -(G+A)@r + (torch.linalg.inv(N)@W@G@N)@r
    else:
        D, A = np.diag(delta), np.diag(alphas)
        return D@x -(G+A)@r + (np.linalg.inv(N)@W@G@N)@r

# Model step function
def step(h,s,e,x,r,W,Ns,beta,sigma,delta,alpha,gamma,useTorch=False):
    if useTorch:
        s_now,e_now,x_now,r_now = (torch.Tensor(s),torch.Tensor(e),
                                   torch.Tensor(x),torch.Tensor(r))
    else:
        s_now,e_now,x_now,r_now = s,e,x,r
    
    s_next = s_now + h*s_dot(s, x, r, beta, alpha, gamma, W, Ns, useTorch)
    e_next = e_now + h*e_dot(s, e, x, beta, sigma, gamma, W, Ns, useTorch)
    x_next = x_now + h*x_dot(e, x, sigma, delta, gamma, W, Ns, useTorch)
    r_next = r_now + h*r_dot(x, r, delta, alpha, gamma, W, Ns, useTorch)
    return s_next, e_next, x_next, r_next


def simulate(h,s0,e0,x0,r0,Ws,beta,sigma,delta,alpha,gammas,Ns,numsteps):
    s,e,x,r = s0,e0,x0,r0
    ss, es, xs, rs = [s], [e], [x], [r]
    for k in range(numsteps):
        W = Ws[k]
        gamma = gammas[k]
        s,e,x,r = step(h,s,e,x,r,W,Ns,beta,sigma,delta,alpha,gamma)

        ss.append(s)
        es.append(e)
        xs.append(x)
        rs.append(r)
    return ss, es, xs, rs

def buildID(ss,es,xs,rs,Ws,gammas,Ns,h,het=False):
    n, T = len(ss[0]), len(ss)
    if het:
        Qs, Deltas = [], []

        for i in range(n):
            Chi_i, Esp_i, Xi_i, Ro_i = np.zeros(((T-1),4)), np.zeros(((T-1),4)), np.zeros(((T-1),4)), np.zeros(((T-1),4))
            Ds_i,De_i,Dx_i,Dr_i = np.zeros(((T-1),1)),np.zeros(((T-1),1)),np.zeros(((T-1),1)),np.zeros(((T-1),1))
            for k in range(T-1):
                s,e,x,r = ss[k],es[k],xs[k],rs[k]
                W, gamma = Ws[k], gammas[k]
                Chi_i[k,0], Chi_i[k,3] = -s[i]*x[i], r[i]
                Esp_i[k,0], Esp_i[k,1] = s[i]*x[i], -e[i]
                Xi_i[k,1], Xi_i[k,2] = e[i], -x[i]
                Ro_i[k,2], Ro_i[k,3] = x[i], -r[i]

                s_flow,e_flow,x_flow,r_flow = 0,0,0,0
                for j in range(n):
                    if j != i:
                        phi = Ns[j]/Ns[i]*W[i,j]*gamma[j]
                        s_flow += phi*s[j]
                        e_flow += phi*e[j]
                        x_flow += phi*x[j]
                        r_flow += phi*r[j]
                        
                Ds_i[k] = ss[k+1][i] - (s[i] - h*gamma[i]*s[i] + s_flow)
                De_i[k] = es[k+1][i] - (e[i] - h*gamma[i]*e[i] + e_flow)
                Dx_i[k] = xs[k+1][i] - (x[i] - h*gamma[i]*x[i] + x_flow)
                Dr_i[k] = rs[k+1][i] - (r[i] - h*gamma[i]*r[i] + r_flow)
            
            Q_i = h*np.block([[Chi_i],[Esp_i],[Xi_i],[Ro_i]])
            Delta_i = np.block([[Ds_i],[De_i],[Dx_i],[Dr_i]])
            Qs.append(Q_i)
            Deltas.append(Delta_i)

        Q = np.zeros((4*n*(T-1), 4*n))
        Delta = np.zeros((4*n*(T-1),1))

        for i in range(len(Qs)):
            Q[4*i*(T-1):4*(i+1)*(T-1),i*4:(i+1)*4] = Qs[i]
            Delta[4*i*(T-1):4*(i+1)*(T-1)] = Deltas[i]
        return Q, Delta

    else:
        Chi, Esp, Xi, Ro = np.zeros((n*(T-1),4)), np.zeros((n*(T-1),4)), np.zeros((n*(T-1),4)), np.zeros((n*(T-1),4))
        Ds, De, Dx, Dr = np.zeros((n*(T-1),1)),np.zeros((n*(T-1),1)),np.zeros((n*(T-1),1)),np.zeros((n*(T-1),1))
        for k in range(T-1):
            s,e,x,r = ss[k],es[k],xs[k],rs[k]
            S = np.diag(s.flatten())
            
            Chi[k*n:(k+1)*n,0] = -(S@x).flatten()
            Chi[k*n:(k+1)*n,3] = r.flatten()
            
            Esp[k*n:(k+1)*n,0] = (S@x).flatten()
            Esp[k*n:(k+1)*n,1] = -e.flatten()
            
            Xi[k*n:(k+1)*n,1] = e.flatten()
            Xi[k*n:(k+1)*n,2] = -x.flatten()
            
            Ro[k*n:(k+1)*n,2] = x.flatten()
            Ro[k*n:(k+1)*n,3] = -r.flatten()

            W, Gamma = Ws[k], np.diag(gammas[k])
            Phi = np.linalg.inv(np.diag(Ns))@W@Gamma@np.diag(Ns)
            Ds[k*n:(k+1)*n] = (ss[k+1] - (np.eye(n)+h*Phi-h*Gamma)@ss[k])
            De[k*n:(k+1)*n] = (es[k+1] - (np.eye(n)+h*Phi-h*Gamma)@es[k])
            Dx[k*n:(k+1)*n] = (xs[k+1] - (np.eye(n)+h*Phi-h*Gamma)@xs[k])
            Dr[k*n:(k+1)*n] = (rs[k+1] - (np.eye(n)+h*Phi-h*Gamma)@rs[k])
        
        Q = h*np.block([[Chi],[Esp],[Xi],[Ro]])
        Delta = np.block([[Ds],[De],[Dx],[Dr]])    
        return Q, Delta

def buildHM(x1,x2,s,beta,sigma,delta,alpha,gammas,W,Ns):
    n = len(Ns)
    B, Sig, D, A = (np.diag(beta.flatten()),np.diag(sigma.flatten()),
                     np.diag(delta.flatten()),np.diag(alpha.flatten()))
    X1, X2, Gamma, N = np.diag(x1.flatten()), np.diag(x2.flatten()), np.diag(gammas.flatten()), np.diag(Ns.flatten())
    S = np.diag(s.flatten())

    Z = np.zeros((n,n))
    

    H = np.block([[B@X1 + Gamma, Z, Z, Z],
                  [Z, Sig + Gamma, Z, Z],
                  [Z, Z, D + Gamma, Z], 
                  [Z, Z, Z, A + Gamma]])
    
    Phi = np.linalg.inv(N)@W@Gamma@N
    
    # M = np.block([[Phi, Z, Z, A], 
    #               [B@X2, Phi, Z, Z], 
    #               [Z, Sig, Phi, Z], 
    #               [Z, Z, D, Phi]])
                  
    M = np.block([[Phi, Z, Z, A], 
                  [Z, Phi, B@S, Z], 
                  [Z, Sig, Phi, Z], 
                  [Z, Z, D, Phi]])
    return H, M

