import numpy as np

# Model dynamics equations
def s_dot(s, x, beta, p_s, W, Ns):
    X, B, N = np.diag(x.flatten()), np.diag(beta), np.diag(Ns)
    P_s = np.diag(p_s)
    return -(B@X+P_s)@s + (np.linalg.inv(N)@W@P_s@N)@s

def e_dot(s, e, x, beta, sigma, p_e, W, Ns):
    S, X, N = np.diag(s.flatten()), np.diag(x.flatten()), np.diag(Ns)
    B, Sig, P_e = np.diag(beta), np.diag(sigma), np.diag(p_e)
    return B@S@x -(Sig+P_e)@e + (np.linalg.inv(N)@W@P_e@N)@e

def x_dot(e, x, sigma, delta, p_x, W, Ns):
    E, Sig, D = np.diag(e.flatten()), np.diag(sigma), np.diag(delta)
    P_x, N = np.diag(p_x), np.diag(Ns)
    return Sig@e -(D+P_x)@x + (np.linalg.inv(N)@W@P_x@N)@x

def r_dot(x, r, delta, p_r, W, Ns):
    X, D = np.diag(x.flatten()), np.diag(delta)
    P_r, N = np.diag(p_r), np.diag(Ns)
    return D@x -P_r@r + (np.linalg.inv(N)@W@P_r@N)@r

# Model step function
def step(h,s,e,x,r,W,Ns,beta,sigma,delta,p_s,p_e,p_x,p_r):
    s_next = s + h*s_dot(s, x, beta, p_s, W, Ns)
    e_next = e + h*e_dot(s, e, x, beta, sigma, p_e, W, Ns)
    x_next = x + h*x_dot(e, x, sigma, delta, p_x, W, Ns)
    r_next = r + h*r_dot(x, r, delta, p_r, W, Ns)
    return s_next, e_next, x_next, r_next

# Compute p_T given the current states
def compute_pT(s,e,x,r,gamma,p_x):
    S,E = np.diag(s.flatten()), np.diag(e.flatten())
    R,Px = np.diag(r.flatten()), np.diag(p_x)
    gamma_vec = np.atleast_2d(gamma).T
    return (np.linalg.inv(S+E+R)@(gamma_vec-Px@x)).flatten()

# Select groups of p_T in a time window for building the classification matrix
def time_var_PT(classify,PT_i,num_win,numdata,numnodes):
    PT_class = np.zeros((numdata,num_win))
    for w in range(num_win):
        PT_i_win = PT_i[:,w*numnodes:(w+1)*numnodes]
        PT_class[:,w] = np.sum(PT_i_win[:,classify], axis=1)
    return PT_class
    
# Build the classification matrix given the state data over time
def buildQ_simple(ss, es, xs, Ns, Ws, het=False, urban=False, cutoff=10e3, t_win=None):
    numdata = len(xs)-1
    numnodes = len(xs[0])
    if het: # for the heterogeneous case...
        Esps, Xis, Deltas = [], [], []
        PTs, Pxs = [], []
        
        for i in range(numnodes): # for each node...
            Esp_i, Xi_i = np.zeros((numdata,3)), np.zeros((numdata,3))
            PT_i, Px_i = np.zeros((numdata,numnodes)), np.zeros((numdata,numnodes))
            Delta_i = np.zeros((2*numdata,1))
            for k in range(numdata): # for each time step...
                s, e, x, W = ss[k], es[k], xs[k], Ws[k]
                e_sum, x_sum = 0, 0
                for j in range(numnodes): # assign the proper vaules associated with p_T and p_x
                    if j != i:
                        PT_i[k,j] = Ns[j]/Ns[i]*W[i,j]*e[j]
                        Px_i[k,j] = Ns[j]/Ns[i]*W[i,j]*x[j]
                    else:
                        PT_i[k,j] = -e[j]
                        Px_i[k,j] = -x[j]
                
                # terms associated with the exposed dynamics
                Esp_i[k,0] = s[i]*x[i]
                Esp_i[k,1] = -e[i]
                # terms associated with the infected dynamics
                Xi_i[k,1] = e[i]
                Xi_i[k,2] = -x[i]
                # compute the difference vector
                Delta_i[k] = es[k+1][i] - e[i]
                Delta_i[k+numdata] = xs[k+1][i] - x[i]
            
            Esps.append(Esp_i)
            Xis.append(Xi_i)
            PTs.append(PT_i)
            Pxs.append(Px_i)
            Deltas.append([Delta_i])
        ##### POSSIBLE ISSUE HERE (maybe, probably not) #####
        if t_win: # if a time window is implemented for p_T
            PTs_win = []
            num_win = int(np.round(numdata/t_win))
            for PT_i in PTs: # for each node
                PT_i_win = np.zeros((numdata,num_win*numnodes))
                for w in range(num_win): # expand PT to learn a separate parameter for each window
                    PT_i_win[w*t_win:(w+1)*t_win,
                             w*numnodes:(w+1)*numnodes] = PT_i[w*t_win:(w+1)*t_win,:]
                
                PTs_win.append(PT_i_win)
            PTs = PTs_win
            Zp_T = np.zeros((numdata,num_win*numnodes))    
        else:
            Zp_T = np.zeros((numdata,numnodes))
        
        Z = np.zeros((2*numdata,3))
        Zp_x = np.zeros((numdata,numnodes))
        EX, P = [], []
        
        if urban: # if an urban/rural split is being implemented...
            urb, rur = [], []
            EspXi_urb, EspXi_rur = [], []
            for i in range(len(Ns)): # separate based on population cutoff
                if Ns[i] >= cutoff: 
                    urb.append(i)
                    EspXi_urb.append([Esps[i]])
                    EspXi_urb.append([Xis[i]])                   
                else:
                    rur.append(i)
                    EspXi_rur.append([Esps[i]])
                    EspXi_rur.append([Xis[i]])
                    
            Delta_UR = []
            PTs_u, Pxs_u, PTs_r, Pxs_r = [], [], [], []
            for u in urb:
                if t_win:
                    PTs_u.append([time_var_PT(urb,PTs[u],num_win,numdata,numnodes)])
                else:
                    PTs_u.append(np.sum(PTs[u][:,urb], axis=1))
                Pxs_u.append(np.sum(Pxs[u][:,urb], axis=1))
                Delta_UR.append(Deltas[u])
            for r in rur:
                if t_win:
                    PTs_r.append([time_var_PT(rur,PTs[r],num_win,numdata,numnodes)])
                else:
                    PTs_r.append(np.sum(PTs[r][:,rur], axis=1))
                Pxs_r.append(np.sum(Pxs[r][:,rur], axis=1))
                Delta_UR.append(Deltas[r])
            
            if t_win:
                ZpT_u, ZpT_r = np.zeros((len(urb)*numdata,num_win)), np.zeros((len(rur)*numdata,num_win))
            else:
                ZpT_u, ZpT_r = np.zeros((len(urb)*numdata,1)), np.zeros((len(rur)*numdata,1))
                
            Zpx_u, Zpx_r = np.zeros((len(urb)*numdata,1)), np.zeros((len(rur)*numdata,1))
            Z_urb, Z_rur = np.zeros((2*len(urb)*numdata,3)), np.zeros((2*len(rur)*numdata,3))
            
            EX = [[np.block(EspXi_urb), Z_urb],
                  [Z_rur, np.block(EspXi_rur)]]
            
            if t_win:
                P_u = [[np.block(PTs_u), Zpx_u],
                       [ZpT_u, np.atleast_2d(np.block(Pxs_u)).T]]
                P_r = [[np.block(PTs_r), Zpx_r],
                       [ZpT_r, np.atleast_2d(np.block(Pxs_r)).T]]
            else:    
                P_u = [[np.atleast_2d(np.block(PTs_u)).T, Zpx_u],
                       [ZpT_u, np.atleast_2d(np.block(Pxs_u)).T]]
                P_r = [[np.atleast_2d(np.block(PTs_r)).T, Zpx_r],
                       [ZpT_r, np.atleast_2d(np.block(Pxs_r)).T]]

            P = [[np.block(P_u), np.zeros(np.block(P_u).shape)],
                 [np.zeros(np.block(P_r).shape), np.block(P_r)]]
            
            Q = np.block([np.block(EX),np.block(P)])
            Delta = np.block(Delta_UR)
            
            return Q, Delta
            
        else:
            for i in range(numnodes):
                P.append([PTs[i], Zp_x])
                P.append([Zp_T, Pxs[i]])
                EspXi_i = np.block([[Esps[i]],[Xis[i]]])
                row_i = []
                for j in range(numnodes):
                    if i == j:
                        row_i.append(EspXi_i)
                    else:
                        row_i.append(Z)
                EX.append(row_i)

            Q = np.block([np.block(EX),np.block(P)])
            Delta = np.block(Deltas)

            return Q, Delta
    
    else: # for the homogeneous case...
        Esp, Xi = np.zeros((numdata*numnodes,5)), np.zeros((numdata*numnodes,5))
        Delta = np.zeros((2*numdata*numnodes,1))
        N = np.diag(Ns)
        for k in range(numdata):
            s, e, x, W = ss[k], es[k], xs[k], Ws[k]
            S = np.diag(s.flatten())
            Esp[k*numnodes:(k+1)*numnodes,0] = (S@x).flatten()
            Esp[k*numnodes:(k+1)*numnodes,1] = -e.flatten()
            Esp[k*numnodes:(k+1)*numnodes,3] = (-e+np.linalg.inv(N)@W@N@e).flatten()
            
            Xi[k*numnodes:(k+1)*numnodes,1] = e.flatten()
            Xi[k*numnodes:(k+1)*numnodes,2] = -x.flatten()
            Xi[k*numnodes:(k+1)*numnodes,4] = (-x+np.linalg.inv(N)@W@N@x).flatten()
            
            Delta[k*numnodes:(k+1)*numnodes ] = es[k+1] - e
            Delta[(k+numdata)*numnodes:(k+numdata+1)*numnodes] = xs[k+1] - x
            
        Q = np.block([[Esp],[Xi]])
        return Q, Delta

def buildQ_px(h, ss, es, xs, Ns, Ws, gammas, het=True):
    numdata = len(xs)-1
    numnodes = len(xs[0])
    if het:
        Esps, Xis, Deltas = [], [], []
        PTs, Pxs = [], []
        
        for i in range(numnodes):
            Esp_i, Xi_i = np.zeros((numdata,3)), np.zeros((numdata,3))
            PT_i, Px_i = np.zeros((numdata,numnodes)), np.zeros((numdata,numnodes))
            Delta_i = np.zeros((2*numdata,1))
            for k in range(numdata):
                s, e, x, W, gamma = ss[k], es[k], xs[k], Ws[k], gammas[k]
                e_sum, x_sum = 0, 0
                for j in range(numnodes):
                    if j != i:
                        PT_i[k,j] = -Ns[j]/Ns[i]*W[i,j]*e[j]/(1-x[j])
                        e_sum += Ns[j]/Ns[i]*W[i,j]*e[j]*gamma[j]/(1-x[j])
                        
                        Px_i[k,j] = Ns[j]/Ns[i]*W[i,j]*x[j]
                    else:
                        PT_i[k,j] = e[j]/(1-x[j])
                        Px_i[k,j] = -x[j]
                
                Esp_i[k,0] = s[i]*x[i]
                Esp_i[k,1] = -e[i]
                
                Xi_i[k,1] = e[i]
                Xi_i[k,2] = -x[i]
                
                Delta_i[k] = es[k+1][i] - e[i] - h*(-gamma[i]*e[i]/(1-x[i])+e_sum)
                Delta_i[k+numdata] = xs[k+1][i] - x[i]
            
            Esps.append(Esp_i)
            Xis.append(Xi_i)
            PTs.append(PT_i)
            Pxs.append(Px_i)
            Deltas.append([Delta_i])
        
        Z = np.zeros((2*numdata,3))
        EX, P = [], []
        
        for i in range(numnodes):
                P.append([PTs[i]])
                P.append([Pxs[i]])
                EspXi_i = np.block([[Esps[i]],[Xis[i]]])
                row_i = []
                for j in range(numnodes):
                    if i == j:
                        row_i.append(EspXi_i)
                    else:
                        row_i.append(Z)
                EX.append(row_i)

        Q = h*np.block([np.block(EX),np.block(P)])
        Delta = np.block(Deltas)
        
        return Q, Delta
    else:
        return None

def compute_param_error(beta, sigma, delta, p_x, estimates, het=False, urban=False):
    if het:
        if urban:
            return None
        else:
            numnodes = len(beta)
            infect_params = estimates[:3*numnodes]
            travel_params = estimates[3*numnodes:]

            beta_hat = infect_params[np.mod(np.arange(infect_params.size),3)==0]
            sigma_hat = infect_params[np.mod(np.arange(infect_params.size),3)==1]
            delta_hat = infect_params[np.mod(np.arange(infect_params.size),3)==2]

            p_x_hat = travel_params[numnodes:]

            b_err = np.sqrt(np.mean((beta - beta_hat)**2))
            s_err = np.sqrt(np.mean((sigma - sigma_hat)**2))
            d_err = np.sqrt(np.mean((delta - delta_hat)**2))
            px_err = np.sqrt(np.mean((p_x-p_x_hat)**2))

            return b_err, s_err, d_err, px_err
    
    else:
        b_err = np.sqrt((beta[0] - estimates[0])**2)
        s_err = np.sqrt((sigma[0] - estimates[1])**2)
        d_err = np.sqrt((delta[0] - estimates[2])**2)
        px_err = np.sqrt((p_x[0] - estimates[4])**2)
        
        return b_err, s_err, d_err, px_err
    
def simulate(h,s0,e0,x0,r0,Ws,beta,sigma,delta,p_x,gammas,Ns,numsteps):
    s,e,x,r = s0,e0,x0,r0
    ss, es, xs, rs = [s], [e], [x], [r]
    p_Ts= []
    for k in range(numsteps):
        W = Ws[k]
        p_T = compute_pT(s,e,x,r,gammas[k],p_x)
        p_Ts.append(p_T)

        p_s, p_e, p_r = np.copy(p_T), np.copy(p_T), np.copy(p_T)
        s,e,x,r = step(h,s,e,x,r,W,Ns,beta,sigma,delta,p_s,p_e,p_x,p_r)

        ss.append(s)
        es.append(e)
        xs.append(x)
        rs.append(r)
    return ss, es, xs, rs