import numpy as np
from scipy.linalg import solve_continuous_are

class LQR:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = None

        #use builtin function to solve ricatti equation
        S = solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        #this solves RK = B^T S, or equivalently K = R^-1 B^T S
        self.K = np.linalg.solve(self.R, self.B.T @ S)


    def compute(self, x):
        
        return -self.K @ x




