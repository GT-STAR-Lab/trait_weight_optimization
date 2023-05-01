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
import exp_setup
import matplotlib.pyplot as plt

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


num_species = 4 #drone,rover,mini-rover,mini-drone
num_tasks = 3  #pick,search for target,move object
num_traits = 9 #speed,footprint,payload,reach,weight,sensing frequency,sensing range,color,battery capacity
num_demo = 1500


def get_random_q(num_species=4):
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

def create_trait_histogram(plt,Q):
    figure, axs = plt.subplots(3,3)
    figure.suptitle("Historgrams of the traits",  fontsize=48)
    for i, ax in enumerate(axs.flat):
        ax.hist(Q[:,:,i], bins=100, density=False,histtype='barstacked', alpha=0.8)

Q = []
for i in range(num_demo):
    Q.append(get_random_q())
Q = np.array(Q)

# plt.rcParams["figure.figsize"] = (18,15)
# create_trait_histogram(plt,Q)
# plt.show()


n_agents_target = np.ones(num_species)*10
y_star = np.array([[ 122.53839519, 0., 119.39044525, 32.15983648, 360.87624558, 2920.25543052, 768.93055775, 0., 0.],
                   [ 120.64147124, 1433.4094241 ,  0.,   0., 358.13547136, 2906.93229221,  766.11629311,   0., 190.],
                   [ 122.31492305, 1441.74082593,  119.06910271,   0., 358.78852644, 0. ,  769.11520065,   0., 193.]])

print(y_star)
def get_X():
    X_test = np.zeros((num_tasks, num_species))
    task_restrict = range(num_tasks)
    for s in range(num_species):
        R = np.random.choice(task_restrict, size=n_agents_target[s].astype(int))
        for m in task_restrict:
            X_test[m, s] = np.sum(R == m)
        X_test = X_test.round()
    X_test = X_test.astype(np.int32)
    return X_test

X = []
for i in range(num_demo):
    X_sol = cp.Variable((num_tasks, num_species), integer=True)

        # minimize trait mismatch
    mismatch_mat = y_star - cp.matmul(X_sol, Q[i])  # trait mismatch matrix


    obj = cp.Minimize(cp.pnorm(mismatch_mat, 2))

    # ensure each agent is only assigned to one task
    constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.array([n_agents_target]).T, X_sol >= 0]

    # solve for X_target
    opt_prob = cp.Problem(obj, constraints)
    opt_prob.solve(solver=cp.CPLEX)
    X_target = X_sol.value
    X.append(X_target)


X = np.array(X)

Y_actual = X@Q

loss = []

for i in range(num_demo):
    l = np.sum((y_star[y_star>0] - Y_actual[i][y_star>0])**2 / (y_star[y_star>0])**2)
    loss.append(l)

print(min(loss))
print(np.mean(loss))
print(max(loss))

ind = np.argpartition(loss, -1000)[-1000:]
print(len(ind))
len(loss)

new_Q = np.delete(Q,ind,axis=0)
new_X = np.delete(X,ind,axis=0)
new_Y_actual = new_X@new_Q

new_norms = []
new_demo_count = new_Q.shape[0]
for i in range(new_demo_count):
    new_norms.append(LA.norm(y_star-new_Y_actual[i], 2))

new_norms = np.array(new_norms) / LA.norm(y_star, 2) #Normalization
print(min(new_norms))
print(np.mean(new_norms))
print(max(new_norms))


norms = []

for i in range(num_demo):
    norms.append(LA.norm(y_star-Y_actual[i], 2))
norms = np.array(norms)/LA.norm(y_star,2)
print(min(norms))
print(np.mean(norms))
print(max(norms))

ind = np.argpartition(norms, -1000)[-1000:]
print(len(ind))

new_Q = np.delete(Q,ind,axis=0)
new_X = np.delete(X,ind,axis=0)
new_Y_actual = new_X@new_Q

new_norms = []
new_demo_count = new_Q.shape[0]

for i in range(new_demo_count):
    new_norms.append(LA.norm(y_star-new_Y_actual[i], 2))

new_norms = np.array(new_norms) / LA.norm(y_star, 2) #Normalization
print(min(new_norms))
print(np.mean(new_norms))
print(max(new_norms))

D = {'X': new_X, 'Q':new_Q, 'Y': new_Y_actual}
