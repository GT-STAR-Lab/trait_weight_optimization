#imports required
from random import seed
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

import plotly.graph_objs as go

from tabulate import tabulate

num_species = 3 #drone,rover,mini-drone
num_tasks = 3  #search,move object, go to goal
num_traits = 3 #speed,payload,sensing range


def calculate_weight(natural_variance, observerd_variance):
    '''
    Calculates the weight given the natural and observed variance.
    Parameters: 
        natural_variance (np.array<float>) [shape-(4,)] : Calculated varaince of Q.
        observed_variance (np.array<float>) [shape-(4,)] : Calculated variance for a task in Y.
    Return: 
        weights (np.array<float>) [shape-4] : Computed weights of the task
    '''

    weights = (natural_variance/2)*np.cos(2*observerd_variance) + 0.5
    return weights

def get_weights(Q_D,Y, natural_weights=True):
    '''
    Calculates the weights given the aggregate Qs and Ys. 
    Parameters: 
        Q_D (np.array<float>) [shape-(num_demo*4,4)] : Aggregate of Qs across the traits (over species). 
        Y (np.array<float>) [shape-(num_demo,3,4)] : Task-trait requirement matrix for every demonstration. 
    Return: 
        weights (np.array<float>) [shape-(3,4)] : Computed weights of all tasks
    '''

    var_N = np.var(Q_D,axis=0)
    var_O = np.var(Y,axis=0)
    cv_N = np.sqrt(var_N)/np.mean(Q_D,axis=0)
    cv_O = np.sqrt(var_O)/np.mean(Y,axis=0)
    natural_var = np.array((1/np.max(cv_N))*cv_N)
    observed_var = cv_O.copy()
    for m in range(num_tasks):
        observed_var[m] = (1/np.max(cv_O[m]))*cv_O[m]
    observed_var = np.array(observed_var)
    if natural_weights == False:
        weight = np.array([calculate_weight((observed_var[m]*0)+0.5,observed_var[m]) for m in range(num_tasks)])
    else:
        weight = np.array([calculate_weight(natural_var,observed_var[m]) for m in range(num_tasks)])
    sums_w = np.sum(weight,axis=1)
    # print("NOT normalized weight matrix is:",weight)
    for m in range(num_tasks):
        weight[m] /= sums_w[m]
    # print("Computer weight matrix is:",weight)

    return weight

def get_weight_error(cur_weight,opt_weight):
    '''
    Calculates the weight error given optimal and current weight.
    Parameters: 
        cur_weight (np.array<float>) [shape-(3,4)] : Computed weights of all tasks.
        opt_weight (np.array<float>) [shape-(3,4)] : Optimal weights for all tasks. 
    Return: 
        error (np.array<float>) [shape-(3,4)] : Computed error per task
    '''
    current_order = np.argsort(cur_weight,axis=1)
    optimal_order = np.argsort(opt_weight,axis=1)
    print("_____________________________")
    print("Ordering of the matrix:",(current_order  == optimal_order))
    print("_____________________________")
    print("Percentage error between initial weights and computed matrix")
    error = []
    for m in range(3):
        error.append(100* abs(cur_weight[m]-opt_weight[m])/opt_weight[m])
    error = np.array(error)
    print(error)
    print("Total error per task:", np.sum(error, axis=1)) 
 
    return error
    

def plot_violin_graph(input, plot_title="Violin plots", n_points=1000):
    '''
    Plot the voilin distribution graph given the input. 
    Parameters: 
        input (np.array<float>) [shape-(N,4)] : N by 4 array where there are more than n_points points.  
        plot_title (String)[optional] : Name for the plot. Default is "Voilin plots"
        n_points (int)[optional] : Number of points to use in the voilin plot. Default is "1000"
    Return: 
        None
    '''
    fs = 10 
    pos = [0]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    axs[0,0].violinplot([input[:,0]], pos, points=n_points, widths=1, showmeans=True, showextrema=True)
    axs[0,0].set_title('Trait 1 violinplot', fontsize=fs)
    axs[0,1].violinplot([input[:,1]], pos, points=n_points, widths=1, showmeans=True, showextrema=True)
    axs[0,1].set_title('Trait 2 violinplot', fontsize=fs)
    axs[1,0].violinplot([input[:,2]], pos, points=n_points, widths=1, showmeans=True, showextrema=True)
    axs[1,0].set_title('Trait 3 violinplot', fontsize=fs)
    axs[1,1].violinplot([input[:,3]], pos, points=n_points, widths=1, showmeans=True, showextrema=True)
    axs[1,1].set_title('Trait 4 violinplot', fontsize=fs)
    fig.suptitle(plot_title)
    fig.subplots_adjust(hspace=0.4)
    plt.show()

def plot_histogram(input,plot_title="Historgrams of the traits",plots_size=2,n_bins=100):
    '''
    Plot the voilin distribution graph given the input. 
    Parameters: 
        input (np.array<float>) [shape-(N,M,4)] : N by M by 4 array where there are more atleast n_bins points.  
        plot_title (String)[optional] : Name for the plot. Default is "Historgrams of the traits"
        plots_size (Tuple<int,int>) : Size of the subplots ,i.e, number of plots needed. Default is (2,2)
        n_bins (int)[optional] : Number of points to use in the voilin plot. Default is "100"
    Return: 
        None
    '''

    figure, axs = plt.subplots(plots_size,plots_size)
    figure.suptitle(plot_title,  fontsize=14)
    for i, ax in enumerate(axs.flat):
        ax.hist(input[:,:,i], bins=n_bins, density=False,histtype='barstacked', alpha=0.8)
    plt.show()


def solve_task_allocation(Q, Y, n_agents_target, opt_weight=None, method=None):
    '''
Computed task allocation matrix    Parameters: 
        Q (np.array<float>) [shape-(4,4)] : Species-trait matrix
        Y (np.array<float>) [shape-(3,4)] : Task-trait matrix     
        n_agents_target (np.array<float>) [shape-(4,)] : No of agents restriction per species. 
        method (string) : Either None, "vector", or "matrix" to describe the optimization technique
    Return: 
        X (np.array<float>) [shape-(3,4)] : Computed task allocation matrix
    '''
    
    X_sol = cp.Variable((num_tasks, num_species), integer=True)

    # minimize trait mismatch
    if opt_weight is None:
        mismatch_mat = cp.pnorm(Y - cp.matmul(X_sol, Q) ,2) # trait mismatch matrix
    else:
        mismatch_mat = 0
        for m in range(num_tasks):
            if method is None:
                weights = np.diag(opt_weight[m])
            else:
                weights = np.diag(np.sqrt(opt_weight[m]))
            mismatch_mat += cp.pnorm(cp.matmul(weights,(Y[m] - cp.matmul(Q.T,X_sol[m]))),2)

    # print(mismatch_mat)

    obj = cp.Minimize(mismatch_mat)
    # ensure each agent is only assigned to one task
    constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.array([n_agents_target]).T, X_sol >= 0]
    
    # solve for X_target
    opt_prob = cp.Problem(obj, constraints)
    opt_prob.solve(solver=cp.CPLEX)
    X_target = X_sol.value
    return X_target



def get_task_allocation_demo(optimal_Y, Q, num_demo, n_agents, opt_weight=None,method=None):
    '''
    Calculates the weight error given optimal and current weight.
    Parameters: 
        optimal_Y (np.array<float>) [shape-(3,4)] : Optimal task-trait matrix   
        num_demo (int) : No of demonstrations.   
        n_agents (int) : No of agents restriction per species. 
        method (string) : Either None, "vector", or "matrix" to describe the optimization technique

    Return: 
        X (np.array<float>) [shape-(num_demo,3,4)] : Computed task allocation matrix
        Q (np.array<float>) [shape-(num_demo,4,4)] : Computed species-trait matrix

    '''
    X = []
    for i in range(num_demo):
        X_tar = solve_task_allocation(Q[i],optimal_Y,n_agents,opt_weight,method)
        X.append(X_tar)
    X = np.array(X)
    return X

def get_random_q(seed_num=None):
    '''
    Computes the distrubtion of species per traits.
    Return: 
        Q (np.array<float>) [shape-(4,4)] : Computed specie-trait matrix
    '''

    rng = rnd.default_rng(seed=seed_num)
    q_1 = np.concatenate([rng.normal(10,2,1),rng.normal(5,1,1),rng.normal(20,3,1)])#speed
    q_2 = np.concatenate([rng.normal(5,1,1),rng.normal(20,3,1),rng.normal(10,2,1)]) #payload
    q_3 = np.concatenate([rng.normal(20,3,1),rng.normal(10,2,1),rng.normal(5,2,1)]) #sensing range
    Q = np.array([q_1, q_2, q_3]).T
    return Q


def get_failed_random_q(seed_num=None):
    '''
    Computes the distrubtion of species per traits.
    Return: 
        Q (np.array<float>) [shape-(4,4)] : Computed specie-trait matrix
    '''

    agent_size = 1
    rng = rnd.default_rng(seed=seed_num)
    q_1 = np.concatenate([rng.normal(10,2, size=agent_size),rng.normal(5, 1, size=agent_size),rng.normal(20, 3, size=agent_size)])#speed
    q_2 = np.concatenate([rng.normal(10,1, size=agent_size),rng.normal(17, 2, size=agent_size),rng.normal(5, 1, size=agent_size)]) #payload
    q_3 = np.concatenate([rng.normal(19,2, size=agent_size),rng.normal(7, 1, size=agent_size),rng.normal(13, 1, size=agent_size)]) #sensing range
    Q = np.array([q_1, q_2, q_3]).T
    return Q

def get_failedQ(num_demo=1500):
    '''
    Parameters: 
        num_demo (int) [optional]: No of demonstrations. Default is 1500
    
    Returns:
        Q (np.array<float>) [shape-(num_demo,4,4)] : Computed species-trait matrix

    '''
    print(num_demo)
    Q = []
    for i in range(num_demo):
        Q.append(get_failed_random_q(i))
    Q = np.array(Q)
    return Q


def get_optimal_Y(X,Q):
    '''
    Calculates the weight error given optimal and current weight.
    Parameters: 
        X (np.array<float>) [shape-(num_demo,3,4)] : Task allocation matrix
        Q (np.array<float>) [shape-(num_demo,4,4)] : Species-trait matrix

    Return: 
        optimal_Y (np.array<float>) [shape-(3,4)] : Optimal task-trait matrix 
    '''    
    Y_test = X@Q
    optimal_Y = np.mean(Y_test,axis=0)

    return optimal_Y


def get_Q(num_demo=1500):
    '''
    Parameters: 
        num_demo (int) [optional]: No of demonstrations. Default is 1500
    
    Returns:
        Q (np.array<float>) [shape-(num_demo,4,4)] : Computed species-trait matrix

    '''
    print("Number of Demonstrations:",num_demo)
    Q = []
    for i in range(num_demo):
        Q.append(get_random_q(i))
    Q = np.array(Q)
    return Q

def get_X_demo(n_agents,num_demo=1500):
    '''
    Parameters: 
        n_agents (int) : No of agents restriction per species.
        num_demo (int) [optional]: No of demonstrations. Default is 1500
    Returns:
        X (np.array<float>) [shape-(num_demo,3,4)] : Computed task allocation matrix
    '''
    X_test = []
    i = 0
    while i < num_demo:
        x_i = np.zeros((num_tasks, num_species))
        task_restrict = range(num_tasks)
        for s in range(num_species):
            R = np.random.choice(task_restrict, size=n_agents[s].astype(int))
            for m in task_restrict:
                x_i[m, s] = np.sum(R == m)
        x_i = x_i.astype(np.int32)
        if len(np.where(~x_i.any(axis=1))[0]) > 0:
            print(x_i)
            continue
        X_test.append(x_i)
        i+=1
    X_test = np.array(X_test)
    return X_test


def refine_demo_elem_distance(optimal_Y, Y, num_demo):
    norms = []
    for i in range(num_demo):
        norms.append(LA.norm(optimal_Y-Y[i], 'fro')) 
    indx = np.argsort(norms)
    return indx

def get_demonstration(num_demos,n_agents_demo,n_agents,opt_weight):
    total_demos = num_demos+num_demos//2
    Q_init = get_Q(total_demos)
    X_init = get_X_demo(n_agents_demo,total_demos)
    Y_init = X_init@Q_init
    y_star = np.mean(Y_init,axis=0)
    print("ystar", y_star)
    ordered_norm = refine_demo_elem_distance(y_star, Y_init, total_demos)
    indx = ordered_norm[:num_demos]
    Q = Q_init[indx]
    X = get_task_allocation_demo(y_star, Q, num_demos, n_agents, opt_weight)
    D = {"X":X, "Q":Q}
    return D,y_star




def plot_Q_3d(Q):

# Configure the trace.
    Species1 = go.Scatter3d(
        x=Q[:,0,0],  # <-- Put your data instead
        y=Q[:,1,0],  # <-- Put your data instead
        z=Q[:,2,0],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Species 1",
    )

    Species2 = go.Scatter3d(
        x=Q[:,0,1],  # <-- Put your data instead
        y=Q[:,1,1],  # <-- Put your data instead
        z=Q[:,2,1],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Species 2",

    )

    Species3 = go.Scatter3d(
        x=Q[:,0,2],  # <-- Put your data instead
        y=Q[:,1,2],  # <-- Put your data instead
        z=Q[:,2,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Species 3",
    )

    data = [Species1,Species2,Species3]

    plot_figure = go.Figure(data=data)

    plot_figure.update_layout(scene = dict(
                        xaxis_title='Trait 1',
                        yaxis_title='Trait 2',
                        zaxis_title='Trait 3'),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10))

    return plot_figure



def plot_Y_3d(Y,y_star):
# Configure the trace.
    task1 = go.Scatter3d(
        x=Y[:,0,0],  # <-- Put your data instead
        y=Y[:,0,1],  # <-- Put your data instead
        z=Y[:,0,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Task 1",
    )

    task2 = go.Scatter3d(
        x=Y[:,1,0],  # <-- Put your data instead
        y=Y[:,1,1],  # <-- Put your data instead
        z=Y[:,1,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Task 2",

    )

    task3 = go.Scatter3d(
        x=Y[:,2,0],  # <-- Put your data instead
        y=Y[:,2,1],  # <-- Put your data instead
        z=Y[:,2,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Task 3",
    )

    y_star_task1 = go.Scatter3d(
        x=[y_star[0,0]],  # <-- Put your data instead
        y=[y_star[0,1]],  # <-- Put your data instead
        z=[y_star[0,2]],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="task 1 y star",
    )

    y_star_task2 = go.Scatter3d(
            x=[y_star[1,0]],  # <-- Put your data instead
            y=[y_star[1,1]],  # <-- Put your data instead
            z=[y_star[1,2]],  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.8,
            },
            name="task 2 y star",
        )

    y_star_task3 = go.Scatter3d(
            x=[y_star[2,0]],  # <-- Put your data instead
            y=[y_star[2,1]],  # <-- Put your data instead
            z=[y_star[2,2]],  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.8,
            },
            name="task 3 y star",
        )


    # Configure the layout.
    # layout = go.Layout(
    #     margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    # )

    data = [task1,task2,task3,y_star_task1,y_star_task2,y_star_task3]

    plot_figure = go.Figure(data=data)
    plot_figure.update_layout(scene = dict(
                        xaxis_title='Trait 1',
                        yaxis_title='Trait 2',
                        zaxis_title='Trait 3'),
                        width=1200,
                        margin=dict(r=20, b=10, l=10, t=10))
    # Render the plot.
    return plot_figure



def plot_QD_3d(QD):

# Configure the trace.
    task1 = go.Scatter3d(
        x=QD[:,0],  # <-- Put your data instead
        y=QD[:,1],  # <-- Put your data instead
        z=QD[:,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 10,
            'opacity': 0.8,
        },
        name="Trait 1",
    )

    data = [task1]

    plot_figure = go.Figure(data=data)

    plot_figure.update_layout(scene = dict(
                        xaxis_title='Trait 1',
                        yaxis_title='Trait 2',
                        zaxis_title='Trait 3'),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10))

    return plot_figure