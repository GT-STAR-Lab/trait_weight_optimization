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

class Experiment:
    traits = ["speed","footprint","payload","reach","weight","sensing frequency","sensing range","color","battery"]

    def __init__(self, species=4,tasks=3,traits=9,demo=1000, agents_per_species=None) -> None:
        self.num_species = species
        self.num_tasks = tasks
        self.num_traits = traits
        self.demo_needed = demo
        self.num_demo = demo+(demo//2)
        self.Q = []
        self.X = []
        self.Y = []
        self.D = {}
        self.n_agents_target = np.ones(self.num_species)*10 if agents_per_species is None else agents_per_species
        self.optimal_Y = np.array([[ 122.53839519, 0., 119.39044525, 32.15983648, 360.87624558, 2920.25543052, 768.93055775, 0., 0.],
                                   [ 120.64147124, 1433.4094241 ,  0.,   0., 358.13547136, 2906.93229221,  766.11629311,   0., 190.],
                                   [ 122.31492305, 1441.74082593,  119.06910271,   0., 358.78852644, 0. ,  769.11520065,   0., 193.]])

        self.create_demonstration()

    def get_random_q(self,num_species=4):
        agent_size = int(num_species/4)
        rng = rnd.default_rng()
        q_1 = np.concatenate([rng.normal(10,1, size=agent_size),rng.normal(5, 1, size=agent_size),rng.normal(7, 1, size=agent_size),rng.normal(15, 2, size=agent_size)])#speed
        q_2 = np.concatenate([rng.normal(90,5, size=agent_size),rng.normal(150, 9, size=agent_size),rng.normal(120, 10, size=agent_size),rng.normal(75, 12, size=agent_size)]) #footprint
        q_3 = np.concatenate([rng.normal(8,2, size=agent_size),rng.normal(5, 1, size=agent_size),rng.normal(16, 1, size=agent_size),rng.normal(7, 2, size=agent_size)]) #payload
        q_4 = np.concatenate([rng.normal(2,1, size=agent_size),rng.normal(3, 1, size=agent_size),rng.normal(2, 1, size=agent_size),rng.normal(3, 2, size=agent_size)]) #reach
        q_5 = rng.normal(np.random.randint(25,30), 1, size=num_species) #weight
        q_6 = rng.normal(np.random.randint(210,230), 15.4, size=num_species) #sensing frequency
        q_7 = rng.normal(np.random.randint(58,60), 0.8, size=num_species) #sensing range
        q_8 = rnd.choice([0,1,2,3,4], num_species) #color
        q_9 = rnd.choice(range(10,20), num_species) #battery
        Q = np.array([q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9]).T
        return Q

    def create_Q(self):
        cur_Q = []
        for i in range(self.num_demo):
            cur_Q.append(self.get_random_q())
        self.Q = np.array(cur_Q)



    def create_trait_histogram(plt,Q):
        figure, axs = plt.subplots(3,3)
        figure.suptitle("Historgrams of the traits",  fontsize=48)
        for i, ax in enumerate(axs.flat):
            ax.hist(Q[:,:,i], bins=100, density=False,histtype='barstacked', alpha=0.8)

    def create_X(self):
        cur_X = []
        for i in range(self.num_demo):
            X_sol = cp.Variable((self.num_tasks, self.num_species), integer=True)

                # minimize trait mismatch
            mismatch_mat = self.optimal_Y - cp.matmul(X_sol, self.Q[i])  # trait mismatch matrix

            obj = cp.Minimize(cp.pnorm(mismatch_mat, 2))

            # ensure each agent is only assigned to one task
            constraints = [cp.matmul(X_sol.T, np.ones([self.num_tasks, 1])) <= np.array([self.n_agents_target]).T, X_sol >= 0]

            # solve for X_target
            opt_prob = cp.Problem(obj, constraints)
            opt_prob.solve(solver=cp.CPLEX)
            X_target = X_sol.value
            cur_X.append(X_target)
        self.X = np.array(cur_X)


    def create_Y(self):
        self.Y = self.X@self.Q

    def refine_demo_elem_distance(self):
        loss = []
        for i in range(self.num_demo):
            l = np.sum((self.optimal_Y[self.optimal_Y>0] - self.Y[i][self.optimal_Y>0])**2 / (self.optimal_Y[self.optimal_Y>0])**2)
            loss.append(l)

        ind = np.argpartition(loss, -self.demo_needed)[-self.demo_needed:]


        self.Q = np.delete(self.Q,ind,axis=0)
        self.X = np.delete(self.X,ind,axis=0)
        self.create_Y()


    def refine_demo_l2_norm(self):
        norms = []
        for i in range(self.num_demo):
            norms.append(LA.norm(self.optimal_Y-self.Y[i], 2)) 

        norms = np.array(norms)/LA.norm(self.optimal_Y,2)

        ind = np.argpartition(norms, -self.demo_needed)[-self.demo_needed:]

        self.Q = np.delete(self.Q,ind,axis=0)
        self.X = np.delete(self.X,ind,axis=0)
        self.create_Y()


    def create_demonstration(self):
        self.create_Q()
        self.create_X()
        self.create_Y()
        self.refine_demo_l2_norm()
        self.D = {'X': self.X, 'Q':self.Q, 'Y': self.Y}
        
    def get_expert_demonstrations(self):
        return self.D

    def get_expert_given_traits(self):
        return self.Q
    
    def get_expert_allocation(self):
        return self.X
    
    def get_agent_limit(self):
        return self.n_agents_target

    def get_optimal_Y(self):
        return self.optimal_Y

# exp = Experiment()
# print(exp.get_expert_demonstrations().keys())