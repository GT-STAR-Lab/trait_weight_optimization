# %% [markdown]
# ### Algorithm Testing
# 

# %%
#imports required
import numpy as np
import matplotlib.pyplot as plt
import plotly
from sim_utilities import *
 
# Configure notebook. Runs on Python 3.6.9
plotly.offline.init_notebook_mode()
%load_ext autoreload
%autoreload 2

# %%
num_species = 3 #drone,rover,mini-drone
num_tasks = 3  #search,move object, go to goal
num_traits = 3 #speed,payload,sensing range
traits = ["speed","payload","sensing range"]
num_demos = 1000

opt_weight = np.array([[0.1,0.8,0.1],
                       [0.8,0.1,0.1],
                       [0.1,0.1,0.8]])

n_agents_demo = np.ones(num_species)*15
n_agents = np.ones(num_species)*15

# %%
D,y_star = get_demonstration(num_demos,n_agents_demo,n_agents,opt_weight)

# %%
X = D["X"]
Q = D["Q"]
Y = X@Q
Q_D = np.concatenate(Q)
weights = get_weights(Q_D,Y)
print(weights)
print(np.argsort(weights) == np.argsort(opt_weight))

# %%
nat_weights = get_weights(Q_D,Y,False)
print(nat_weights)

# %%
plt.rcParams["figure.figsize"] = (3,6)
figure, axs = plt.subplots(3,squeeze=False)
figure.suptitle("Histograms of Trait distribution across species",  fontsize=14)
for i, ax in enumerate(axs.flat):
    ax.hist(Q[:,:,i], bins=150, density=False,histtype='barstacked', alpha=0.8)
plt.show()

plot_figure = plot_Y_3d(Y, np.mean(Y,axis=0))
plot_figure.show()

# %%
plot_QD_3d(Q_D).show()

# %%
Y_mean = Y.mean(axis=0)
baseline = []
algo1 = []
algo_no_nat = []

for i in range(100):
    new_Q = get_random_q(i)
    X_per_task = solve_task_allocation(new_Q,Y_mean,n_agents)
    n_Y = X_per_task@new_Q
    baseline.append(n_Y)
    # baseline.append(LA.norm(y_star-n_Y, 2)/LA.norm(n_Y, 2))
    X_per_task_1 = solve_task_allocation(new_Q,Y_mean,n_agents,weights)
    n_Y_1 = X_per_task_1@new_Q
    algo1.append(n_Y_1)
    # algo1.append(LA.norm(y_star-n_Y_1, 2)/LA.norm(n_Y_1, 2))
    no_X_per_task_1 = solve_task_allocation(new_Q,Y_mean,n_agents,nat_weights)
    no_n_Y_1 = no_X_per_task_1@new_Q
    algo_no_nat.append(no_n_Y_1)

baseline = np.array(baseline)
algo1 = np.array(algo1)
algo_no_nat = np.array(algo_no_nat)

# %%
norm_base = []
norm_algo = []
no_norm_algo =[]
for i in range(100):
    norm_base.append(LA.norm(opt_weight*(y_star-baseline[i]),axis=1)/LA.norm(opt_weight*(y_star),axis=1))
    norm_algo.append(LA.norm(opt_weight*(y_star-algo1[i]),axis=1)/LA.norm(opt_weight*(y_star),axis=1))
    no_norm_algo.append(LA.norm(opt_weight*(y_star-algo_no_nat[i]),axis=1)/LA.norm(opt_weight*(y_star),axis=1))

norm_base = np.array(norm_base)
norm_algo = np.array(norm_algo)
no_norm_algo = np.array(no_norm_algo)

# # %%
# y = norm_base[:,0]
# y2 = norm_algo[:,0]
# y3 = no_norm_algo[:,0]


# trace = go.Violin(
#     y=y, 
#     name = 'Without Preference',
#     marker = dict(color = 'rgb(214,12,140)'),box_visible=True,
#                             meanline_visible=True)

# trace2 = go.Violin(
#     y=y3,
#     name = 'With Observed Variation',
#     marker = dict(color = 'rgb(12,140,214)'),box_visible=True,
#                             meanline_visible=True)

# trace3 = go.Violin(
#     y=y2,
#     name = 'Our Method (With Inherent Diversity and Observed Variation)',
#     marker = dict(color = 'rgb(12,214,140)'),box_visible=True,
#                             meanline_visible=True)

# layout = go.Layout(
#     height=1000,
#     yaxis=dict(
#         title='<b>Weighted Trait Mismatch Error</b>',
#     ),
#     font=dict(
#         size=18,
#     ),
#     title={
#         'text': '<b>Task 1</b>',
#         'y':0.95,
#         'x':0.35,
#         'xanchor': 'center',
#         'yanchor': 'top'}
# )

# data = [trace,trace2,trace3]
# fig= go.Figure(data=data, layout=layout)
# plotly.offline.iplot(fig)

# # %%
# y = norm_base[:,1]
# y2 = norm_algo[:,1]
# y3 = no_norm_algo[:,1]


# trace = go.Violin(
#     y=y, 
#     name = 'Without Preference',
#     marker = dict(color = 'rgb(214,12,140)'),box_visible=True,
#                             meanline_visible=True)

# trace2 = go.Violin(
#     y=y3,
#     name = 'With Observed Variation',
#     marker = dict(color = 'rgb(12,140,214)'),box_visible=True,
#                             meanline_visible=True)

# trace3 = go.Violin(
#     y=y2,
#     name = 'Our Method (With Inherent Diversity and Observed Variation)',
#     marker = dict(color = 'rgb(12,214,140)'),box_visible=True,
#                             meanline_visible=True)

# layout = go.Layout(
#     height=1000,
#     yaxis=dict(
#         title='<b>Weighted Trait Mismatch Error</b>',
#     ),
#     font=dict(
#         size=18,
#     ),
#     title={
#         'text': '<b>Task 2</b>',
#         'y':0.95,
#         'x':0.35,
#         'xanchor': 'center',
#         'yanchor': 'top'}
# )

# data = [trace,trace2,trace3]
# fig= go.Figure(data=data, layout=layout)
# plotly.offline.iplot(fig)

# # %%
# y = norm_base[:,2]
# y2 = norm_algo[:,2]
# y3 = no_norm_algo[:,2]


# trace = go.Violin(
#     y=y, 
#     name = 'Without Preference',
#     marker = dict(color = 'rgb(214,12,140)'),box_visible=True,
#                             meanline_visible=True)

# trace2 = go.Violin(
#     y=y3,
#     name = 'With Observed Variation',
#     marker = dict(color = 'rgb(12,140,214)'),box_visible=True,
#                             meanline_visible=True)

# trace3 = go.Violin(
#     y=y2,
#     name = 'Our Method (With Inherent Diversity and Observed Variation)',
#     marker = dict(color = 'rgb(12,214,140)'),box_visible=True,
#                             meanline_visible=True)

# layout = go.Layout(
#     height=1000,
#     yaxis=dict(
#         title='<b>Weighted Trait Mismatch Error</b>',
#     ),
#     font=dict(
#         size=18,
#     ),
#     title={
#         'text': '<b>Task 3</b>',
#         'y':0.95,
#         'x':0.35,
#         'xanchor': 'center',
#         'yanchor': 'top'}
# )

# data = [trace,trace2,trace3]
# fig= go.Figure(data=data, layout=layout)
# plotly.offline.iplot(fig)

# %%
from scipy import stats

stats.kruskal(baseline[:,0],algo_no_nat[:,0],algo1[:,0]),stats.kruskal(baseline[:,1],algo_no_nat[:,1],algo1[:,1]),stats.kruskal(baseline[:,2],algo_no_nat[:,2],algo1[:,2])

stats.mannwhitneyu(algo_no_nat[:,0],baseline[:,0]),stats.mannwhitneyu(algo_no_nat[:,1],baseline[:,1]),stats.mannwhitneyu(algo_no_nat[:,2],baseline[:,2])

stats.mannwhitneyu(algo_no_nat[:,0],algo1[:,0]),stats.mannwhitneyu(algo_no_nat[:,1],algo1[:,1]),stats.mannwhitneyu(algo_no_nat[:,2],algo1[:,2])

stats.mannwhitneyu(algo1[:,0],baseline[:,0]),stats.mannwhitneyu(algo1[:,1],baseline[:,1]),stats.mannwhitneyu(algo1[:,2],baseline[:,2])

