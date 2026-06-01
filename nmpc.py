import numpy as np
from scipy.optimize import minimize


class nMPC:



    def __init__(self,dyn,N,dt_mpc,Q,R,Q_f,F_max):
        
        #initialize vars

        self.dyn = dyn
        self.N = N
        self.dt = dt_mpc
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.F_max = F_max
        self.warm_start = np.zeros(N)
    def rk4(self,s,f_val):
        #s: config vector [xc, θ1, θ2, ẋc, θ̇1, θ̇2]

        #f_val: force

        dt = self.dt
        k1 = self.dyn(s, f_val)
        k2 = self.dyn(s + 0.5*dt*k1, f_val)
        k3 = self.dyn(s + 0.5*dt*k2, f_val)
        k4 = self.dyn(s + dt*k3, f_val)
        return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def cost(self, U,x0):

        #assume ref = 0
        cost = 0.0
        s=x0.copy()
        for k in range(self.N):
            #loop over horizon
            cost += s @ self.Q @ s + U[k] * self.R[0,0] * U[k]
            s = self.rk4(s, U[k])
        # Terminal cost
        cost += s @ self.Q_f @ s
        
        return cost
    def compute(self,x0):

        bounds = [(-self.F_max, self.F_max)] * self.N

        result = minimize(
            self.cost,
            self.warm_start,
            args=(x0,),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        #warm start and bump 0-index to last
        self.warm_start = np.append(result.x[1:], result.x[-1])
        
        #get first control input
        return result.x[0]
