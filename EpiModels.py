import numpy as np

class SIRS:
    def __init__(self, beta, delta, alpha, Ns):
      self.beta = beta
      self.delta = delta
      self.alpha = alpha
      self.Ns = Ns

    def s_dot(self, s, x, r, gammas, W):
        X, N, G = np.diag(x.flatten()), np.diag(self.Ns), np.diag(gammas)
        B, A = np.diag(self.beta), np.diag(self.alpha)
        return -(B@X+G)@s + (np.linalg.inv(N)@W@G@N)@s + A@r

    def x_dot(self, s, x, gammas, W):
        X, N, G = np.diag(x.flatten()), np.diag(self.Ns), np.diag(gammas)
        B, D = np.diag(self.beta), np.diag(self.delta)
        return B@X@s -(D+G)@x + (np.linalg.inv(N)@W@G@N)@x

    def r_dot(self, x, r, gammas, W):
        N, G = np.diag(self.Ns), np.diag(gammas)
        D, A = np.diag(self.delta), np.diag(self.alpha)
        return D@x -(G+A)@r + (np.linalg.inv(N)@W@G@N)@r

    # Model step function
    def step(self,h,s,x,r,W,gamma):
        s_now, x_now, r_now = s, x, r
        
        s_next = s_now + h*self.s_dot(s, x, r, gamma, W)
        x_next = x_now + h*self.x_dot(s, x, gamma, W)
        r_next = r_now + h*self.r_dot(x, r, gamma, W)
        return s_next, x_next, r_next


    def simulate(self,h,s0,x0,r0,Ws,gammas,numsteps):
        s, x, r = s0, x0, r0
        ss, xs, rs = [s], [x], [r]
        for k in range(numsteps):
            W = Ws[k]
            gamma = gammas[k]
            s,x,r = self.step(h,s,x,r,W,gamma)

            ss.append(s)
            xs.append(x)
            rs.append(r)
        return ss, xs, rs

class SIS:
    def __init__(self, beta, alpha, Ns):
      self.beta = beta
      self.alpha = alpha
      self.Ns = Ns

    def s_dot(self, s, x, gammas, W):
        X, N, G = np.diag(x.flatten()), np.diag(self.Ns), np.diag(gammas)
        B, A = np.diag(self.beta), np.diag(self.alpha)
        return -(B@X+G)@s + (np.linalg.inv(N)@W@G@N)@s + A@x

    def x_dot(self, s, x, gammas, W):
        X, N, G = np.diag(x.flatten()), np.diag(self.Ns), np.diag(gammas)
        B, A = np.diag(self.beta), np.diag(self.alpha)
        return B@X@s -(A+G)@x + (np.linalg.inv(N)@W@G@N)@x


    # Model step function
    def step(self,h,s,x,W,gamma):
        s_now, x_now = s, x,
        
        s_next = s_now + h*self.s_dot(s, x, gamma, W)
        x_next = x_now + h*self.x_dot(s, x, gamma, W)
        return s_next, x_next


    def simulate(self,h,s0,x0,Ws,gammas,numsteps):
        s, x = s0, x0
        ss, xs = [s], [x]
        for k in range(numsteps):
            W = Ws[k]
            gamma = gammas[k]
            s,x = self.step(h,s,x,W,gamma)

            ss.append(s)
            xs.append(x)
        return ss, xs