import numpy as np

# Model dynamics equations
def s_dot(s, e, x, r, beta_e, beta_x, alpha, W):
    Be, Bx, A = np.diag(beta_e), np.diag(beta_x), np.diag(alpha)
    Iota = np.diag((Be@W@e).flatten()) + np.diag((Bx@W@x).flatten())
    return -Iota@s + A@r

def e_dot(s, e, x, beta_e, beta_x, sigma, W):
    Be, Bx = np.diag(beta_e), np.diag(beta_x)
    Iota = np.diag((Be@W@e).flatten()) + np.diag((Bx@W@x).flatten())
    Sig  = np.diag(sigma)
    return Iota@s - Sig@e

def x_dot(e, x, sigma, delta):
    Sig, D = np.diag(sigma), np.diag(delta)
    return Sig@e -D@x

def r_dot(x, r, delta, alphas):
    D, A = np.diag(delta), np.diag(alphas)
    return D@x -A@r

# Model step function
def step(h,s,e,x,r,W,beta_e,beta_x,sigma,delta,alpha):
    s_now,e_now,x_now,r_now = s,e,x,r
    
    s_next = s_now + h*s_dot(s, e, x, r, beta_e, beta_x, alpha, W)
    e_next = e_now + h*e_dot(s, e, x, beta_e, beta_x, sigma, W)
    x_next = x_now + h*x_dot(e, x, sigma, delta)
    r_next = r_now + h*r_dot(x, r, delta, alpha)
    return s_next, e_next, x_next, r_next


def simulate(h,s0,e0,x0,r0,Ws,beta_e,beta_x,sigma,delta,alpha,numsteps):
    s,e,x,r = s0,e0,x0,r0
    ss, es, xs, rs = [s], [e], [x], [r]
    for k in range(numsteps):
        W = Ws[k]
        s,e,x,r = step(h,s,e,x,r,W,beta_e,beta_x,sigma,delta,alpha)

        ss.append(s)
        es.append(e)
        xs.append(x)
        rs.append(r)
    return ss, es, xs, rs

def buildID(ss,es,xs,rs,Ws,h,het=False):
    n, T = len(ss[0]), len(ss)
    if het:
        Qs, Deltas = [], []

        for i in range(n):
            Chi_i, Esp_i, Xi_i, Ro_i = np.zeros(((T-1),5)), np.zeros(((T-1),5)), np.zeros(((T-1),5)), np.zeros(((T-1),5))
            Ds_i,De_i,Dx_i,Dr_i = np.zeros(((T-1),1)),np.zeros(((T-1),1)),np.zeros(((T-1),1)),np.zeros(((T-1),1))
            for k in range(T-1):
                s,e,x,r,W = ss[k],es[k],xs[k],rs[k],Ws[k]
                
                e_flow, x_flow = 0,0
                for j in range(n):
                    if j != i:
                        e_flow += W[i,j]*e[j]
                        x_flow += W[i,j]*x[j]
                
                Chi_i[k,0], Chi_i[k,1], Chi_i[k,4] = -s[i]*e_flow, -s[i]*x_flow, r[i]
                Esp_i[k,0], Esp_i[k,1], Esp_i[k,2] = s[i]*e_flow, s[i]*x_flow, -e[i]
                Xi_i[k,2], Xi_i[k,3] = e[i], -x[i]
                Ro_i[k,3], Ro_i[k,4] = x[i], -r[i]
                        
                Ds_i[k] = ss[k+1][i] - s[i]
                De_i[k] = es[k+1][i] - e[i]
                Dx_i[k] = xs[k+1][i] - x[i]
                Dr_i[k] = rs[k+1][i] - r[i]
            
            Q_i = h*np.block([[Chi_i],[Esp_i],[Xi_i],[Ro_i]])
            Delta_i = np.block([[Ds_i],[De_i],[Dx_i],[Dr_i]])
            Qs.append(Q_i)
            Deltas.append(Delta_i)

        Q = np.zeros((4*n*(T-1), 5*n))
        Delta = np.zeros((4*n*(T-1),1))

        for i in range(len(Qs)):
            Q[4*i*(T-1):4*(i+1)*(T-1),i*5:(i+1)*5] = Qs[i]
            Delta[4*i*(T-1):4*(i+1)*(T-1)] = Deltas[i]
        return Q, Delta

    else:
        Chi, Esp, Xi, Ro = np.zeros((n*(T-1),5)), np.zeros((n*(T-1),5)), np.zeros((n*(T-1),5)), np.zeros((n*(T-1),5))
        Ds, De, Dx, Dr = np.zeros((n*(T-1),1)),np.zeros((n*(T-1),1)),np.zeros((n*(T-1),1)),np.zeros((n*(T-1),1))
        for k in range(T-1):
            s,e,x,r,W = ss[k],es[k],xs[k],rs[k],Ws[k]
            S = np.diag(s.flatten())
            
            Chi[k*n:(k+1)*n,0] = -(S@W@e).flatten()
            Chi[k*n:(k+1)*n,1] = -(S@W@x).flatten()
            Chi[k*n:(k+1)*n,4] = r.flatten()
            
            Esp[k*n:(k+1)*n,0] = (S@W@e).flatten()
            Esp[k*n:(k+1)*n,1] = (S@W@x).flatten()
            Esp[k*n:(k+1)*n,2] = -e.flatten()
            
            Xi[k*n:(k+1)*n,2] = e.flatten()
            Xi[k*n:(k+1)*n,3] = -x.flatten()
            
            Ro[k*n:(k+1)*n,3] = x.flatten()
            Ro[k*n:(k+1)*n,4] = -r.flatten()

            Ds[k*n:(k+1)*n] = (ss[k+1] - s)
            De[k*n:(k+1)*n] = (es[k+1] - e)
            Dx[k*n:(k+1)*n] = (xs[k+1] - x)
            Dr[k*n:(k+1)*n] = (rs[k+1] - r)
        
        Q = h*np.block([[Chi],[Esp],[Xi],[Ro]])
        Delta = np.block([[Ds],[De],[Dx],[Dr]])    
        return Q, Delta

def buildID_noE(ss,es,xs,rs,Ws,h,het=False):
    n, T = len(ss[0]), len(ss)
    if het:
        Qs, Deltas = [], []

        for i in range(n):
            Chi_i, Esp_i, Xi_i, Ro_i = np.zeros(((T-1),4)), np.zeros(((T-1),4)), np.zeros(((T-1),4)), np.zeros(((T-1),4))
            Ds_i,De_i,Dx_i,Dr_i = np.zeros(((T-1),1)),np.zeros(((T-1),1)),np.zeros(((T-1),1)),np.zeros(((T-1),1))
            for k in range(T-1):
                s,e,x,r,W = ss[k],es[k],xs[k],rs[k],Ws[k]
                
                e_flow, x_flow = 0,0
                for j in range(n):
                    if j != i:
                        e_flow += W[i,j]*e[j]
                        x_flow += W[i,j]*x[j]
                
                Chi_i[k,0], Chi_i[k,3] = -s[i]*x_flow, r[i]
                Esp_i[k,0], Esp_i[k,1] = s[i]*x_flow, -e[i]
                Xi_i[k,1], Xi_i[k,2] = e[i], -x[i]
                Ro_i[k,2], Ro_i[k,3] = x[i], -r[i]
                        
                Ds_i[k] = ss[k+1][i] - s[i]
                De_i[k] = es[k+1][i] - e[i]
                Dx_i[k] = xs[k+1][i] - x[i]
                Dr_i[k] = rs[k+1][i] - r[i]
            
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
            s,e,x,r,W = ss[k],es[k],xs[k],rs[k],Ws[k]
            S = np.diag(s.flatten())
            
            Chi[k*n:(k+1)*n,0] = -(S@W@x).flatten()
            Chi[k*n:(k+1)*n,3] = r.flatten()
            
            Esp[k*n:(k+1)*n,0] = (S@W@x).flatten()
            Esp[k*n:(k+1)*n,1] = -e.flatten()
            
            Xi[k*n:(k+1)*n,1] = e.flatten()
            Xi[k*n:(k+1)*n,2] = -x.flatten()
            
            Ro[k*n:(k+1)*n,2] = x.flatten()
            Ro[k*n:(k+1)*n,3] = -r.flatten()

            Ds[k*n:(k+1)*n] = (ss[k+1] - s)
            De[k*n:(k+1)*n] = (es[k+1] - e)
            Dx[k*n:(k+1)*n] = (xs[k+1] - x)
            Dr[k*n:(k+1)*n] = (rs[k+1] - r)
        
        Q = h*np.block([[Chi],[Esp],[Xi],[Ro]])
        Delta = np.block([[Ds],[De],[Dx],[Dr]])    
        return Q, Delta


