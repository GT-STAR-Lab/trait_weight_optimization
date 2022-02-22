import numpy as np
import numpy.random as rnd
import scipy
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from numpy import linalg as LA
import cvxpy as cp
import matplotlib.pyplot as plt

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from exp_setup import Experiment

num_species = 4 #drone,rover,mini-rover,mini-drone
num_tasks = 3  #pick,search for target,move object
num_traits = 9 #speed,footprint,payload,reach,weight,sensing frequency,sensing range,color,battery capacity

def solve_task_allocation(n_agents_target, new_Q, Y_mean):

    X_sol = cp.Variable((num_tasks, num_species), integer=True)

        # minimize trait mismatch
    mismatch_mat = Y_mean - cp.matmul(X_sol, new_Q)  # trait mismatch matrix


    obj = cp.Minimize(cp.pnorm(mismatch_mat, 2))
    # ensure each agent is only assigned to one task
    constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.array([n_agents_target]).T, X_sol >= 0]
    
    # solve for X_target
    opt_prob = cp.Problem(obj, constraints)
    opt_prob.solve(solver=cp.CPLEX)
    X_target = X_sol.value
    return X_target

def solve_task_allocation_per_task(n_agents_target, new_Q, Y_mean):
    X = []
    for m in range(num_tasks):
        X_sol = cp.Variable((num_species), integer=True)

        # minimize trait mismatch
        mismatch_mat = Y_mean[m] - cp.matmul(new_Q.T,X_sol)  # trait mismatch matrix


        obj = cp.Minimize(cp.pnorm(mismatch_mat, 2))
        # ensure each agent is only assigned to one task
        constraints = [X_sol <= np.array(n_agents_target), X_sol >= 0]
        
        # solve for X_target
        opt_prob = cp.Problem(obj, constraints)
        opt_prob.solve(solver=cp.CPLEX)
        X_target = X_sol.value
        X.append(X_target)
    return X
