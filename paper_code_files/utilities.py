from curses import termattrs
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import cvxpy as cp
import time
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
from sklearn.model_selection import KFold,GroupKFold


INFO_COLS = ['short_name','nationality','team_position','overall']
POSITIONS = np.array(["Forward", "Midfield", "Defense", "GK"])

def get_positions():
  return POSITIONS

def get_info_cols():
  return INFO_COLS

def get_allocation(opt_weight,cur_Q, Y_star,team_requirement=None):
  num_species = len(cur_Q)
  num_tasks = len(Y_star)

  if team_requirement is None:
    team_requirement = np.ones(num_tasks)
  target_Y = Y_star * team_requirement[:, np.newaxis]
  n_agents_target = np.ones(num_species)
  
  # print(num_species,num_tasks)
  X_sol = cp.Variable((num_tasks, num_species), integer=True)
    # minimize trait mismatch
  if opt_weight is None:
      mismatch_mat = cp.pnorm(target_Y - cp.matmul(X_sol,cur_Q) ,2) # trait mismatch matrix
  else:
      mismatch_mat = 0
      for m in range(num_tasks):
          weights = np.diag(opt_weight[m])
          mismatch_mat += cp.pnorm(cp.matmul(weights,(target_Y[m] - cp.matmul(cur_Q.T,X_sol[m]))),2)

  # print(mismatch_mat)

  obj = cp.Minimize(mismatch_mat)
  # ensure each agent is only assigned to one task
  constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.array([n_agents_target]).T,
                    cp.matmul(X_sol, np.ones([num_species, 1])) == np.array([team_requirement]).T, 
                       X_sol >= 0]

  # solve for X_target
  opt_prob = cp.Problem(obj, constraints)
  opt_prob.solve(solver=cp.CPLEX)
  X_target = X_sol.value
  return X_target

def get_score(allocation, actual_pos, percentage=False):
  rows_base, idx_base = np.where(allocation>0.5)
  score = (POSITIONS[rows_base].astype(str) == actual_pos[idx_base].astype(str)).sum()
  if percentage:
    return 100*score/len(actual_pos)
  return score

def load_experiment_setup_to_file(dataset, folder, formation):
    players = formation.sum()
    f = open(folder+"/data.txt")
    iterations = int(f.readline().strip())
    info = {}
    teams = {}
    for i in range(4):
        filename = folder+"/train_idx"+str(i)+".txt"
        train_idx = np.loadtxt(filename)
        train_data = dataset.loc[train_idx]
        Q = get_Q(train_data)
        Y = get_Y(train_data)
        info[i] = {}
        info[i]['Q'] = Q
        info[i]['Y'] = Y
        line1 = f.readline().strip()[1:].replace(" ",",")
        line2 = f.readline().strip()[:-1].replace(" ",",")
        countries = list(eval((line1)+","+(line2)))
        for cur_c in (countries):
            teams[cur_c] = {}
            for iter in range(iterations):
                teams[cur_c][iter] = {}
                target_Q = []
                for t in range(players):
                    line = f.readline()
                    # print(line)
                    target_Q.append(list(eval(line)))
                target_Q = np.array(target_Q)
                if players == 4:
                    actual_pos = list(eval((f.readline().strip()[1:-1]).replace(" ",",")))
                else:
                    line1 = f.readline()[1:].strip().replace(" ",",")
                    line2 = f.readline()[:-2].strip().replace(" ",",")
                    actual_pos = list(eval((line1)+","+(line2)))
                teams[cur_c][iter]['Q'] = target_Q
                teams[cur_c][iter]['pos'] = actual_pos
    f.close()
    return info,teams

def write_experiment_setup_to_file(dataset,team_formation, iteration_count=10, path="experiment"):
    print(path)
    unique_vals = dataset['nationality'].unique()
    nations = dataset['nationality'].replace(to_replace=unique_vals,
            value= list(range(len(unique_vals))), inplace=False)
    kf = GroupKFold(4)
    count = 0
    f = open(path+'/data.txt', 'w+') 
    print(iteration_count, file=f)
    for train_index, test_index in kf.split(dataset,groups=nations):
        test_data = dataset.loc[test_index]
        filename = path+"/train_idx"+str(count)+".txt"
        np.savetxt(filename, train_index, delimiter=',')
        print(test_data['nationality'].unique(), file=f)
        for cur_country in test_data['nationality'].unique(): 
            for iter in range(iteration_count):
                team = get_target_Q(test_data, cur_country, team_formation)
                if team is None:
                    print(cur_country)
                    continue
                target_Q = np.array(team.drop(columns=get_info_cols()))
                actual_pos = np.array([str(pos) for pos in team['team_position']])
                np.savetxt(f, target_Q,delimiter=',')
                print(actual_pos,file=f)
        count+=1
    f.close()
    return True




def run_experiment(info,teams, team_formation, num_zeros=0, balance=False, normalize=False, use_natural_var=False):
  scores = {}

        
  for i in range(4):
    Q = info[i]['Q']
    Y = info[i]['Y']
    Y_star = get_y_star(Y)
    for cur_country in teams.keys(): 
      scores[cur_country] = {}
      for iter in teams[cur_country].keys():
        target_Q = teams[cur_country][iter]['Q']
        actual_pos =  np.array((teams[cur_country][iter]['pos'])).astype(str)
        weights= get_weights(Q,Y, num_zeros, balance, normalize, use_natural_var)
        # weights = np.array([forward,mid,defend,gk])
        start_time = time.time()
        X = get_allocation(weights, target_Q,Y_star, team_formation)
        time_taken = time.time()-start_time
        algo_score= get_score(X, actual_pos)
        scores[cur_country][iter] = {}
        scores[cur_country][iter]['score'] = int(algo_score)
        scores[cur_country][iter]['time'] = time_taken
  return scores

def run_baseline(info,teams, team_formation):
  scores = {}
  for i in range(4):
    Q = info[i]['Q']
    Y = info[i]['Y']
    Y_star = get_y_star(Y)
    for cur_country in teams.keys(): 
      scores[cur_country] = {}
      for iter in teams[cur_country].keys():
        target_Q = teams[cur_country][iter]['Q']
        scores[cur_country][iter] = {}
        actual_pos =  np.array((teams[cur_country][iter]['pos'])).astype(str)
        start_time = time.time()
        X = get_allocation(None, target_Q,Y_star, team_formation)
        time_taken = time.time()-start_time
        algo_score = get_score(X, actual_pos)
        scores[cur_country][iter]["score"] = int(algo_score)
        scores[cur_country][iter]["time"] = time_taken
  return scores

def create_dataset(filename, normalize=False):
    df = pd.read_csv(filename)
    # print("Dataset has ",len(df), "players with", len(df.columns)-4," traits and 4 columns of information")
    new_pd = df.fillna("NA")
    new_pd = new_pd.replace('RF', 'Forward')
    new_pd = new_pd.replace('CF', 'Forward')
    new_pd = new_pd.replace('LF', 'Forward')
    new_pd = new_pd.replace('ST', 'Forward')
    new_pd = new_pd.replace('LS', 'Forward')
    new_pd = new_pd.replace('RS', 'Forward')

    new_pd = new_pd.replace('LWB', 'Defense')
    new_pd = new_pd.replace('LCB', 'Defense')
    new_pd = new_pd.replace('LB', 'Defense')
    new_pd = new_pd.replace('RWB', 'Defense')
    new_pd = new_pd.replace('RB', 'Defense')
    new_pd = new_pd.replace('RCB', 'Defense')
    new_pd = new_pd.replace('LCB', 'Defense')
    new_pd = new_pd.replace('CB', 'Defense')
    new_pd = new_pd.replace('RW', 'Defense')
    new_pd = new_pd.replace('LW', 'Defense')

    new_pd = new_pd.replace('CAM', 'Midfield')
    new_pd = new_pd.replace('CM', 'Midfield')
    new_pd = new_pd.replace('RAM', 'Midfield')
    new_pd = new_pd.replace('LAM', 'Midfield')
    new_pd = new_pd.replace('RCM', 'Midfield')
    new_pd = new_pd.replace('LCM', 'Midfield')
    new_pd = new_pd.replace('CDM', 'Midfield')
    new_pd = new_pd.replace('RDM', 'Midfield')
    new_pd = new_pd.replace('LDM', 'Midfield')
    new_pd = new_pd.replace('LM', 'Midfield')
    new_pd = new_pd.replace('RM', 'Midfield')

    new_pd = new_pd.replace('SUB', 'Other')
    new_pd = new_pd.replace('RES', 'Other')
    new_pd = new_pd.replace('NA', 'Other')
    new_pd.drop(new_pd.loc[new_pd['team_position']=='Other'].index, inplace=True)

    # print("POSITIONS are ", new_pd['team_position'].unique())
    
    dataset = new_pd
    Q = new_pd.drop(columns=INFO_COLS)
    if normalize:
        Q =(Q-Q.min())/(Q.max()-Q.min())
        labels = new_pd[INFO_COLS]
        dataset = pd.concat([labels,Q],axis=1, join='inner')

    return dataset
  
def get_Q(dataset):
  Q = dataset.drop(columns=INFO_COLS)
  return Q

def get_Y(dataset):
  Y = []
  for pos in POSITIONS:
      temp_df = dataset[dataset['team_position']==pos]
      cur_Y = temp_df.drop(columns=INFO_COLS)
      Y.append(cur_Y)
  return Y
  
def get_y_star(Y):
    Y_star = []
    num_tasks = len(Y)
    for i in range(num_tasks):
        mean_y = (np.mean(Y[i],axis=0))
        Y_star.append(mean_y)
    Y_star = np.array(Y_star)
    return Y_star

def calculate_weight(natural_variance, observerd_variance):
    weights = (natural_variance/2)*np.cos(2*observerd_variance-0.5) + 0.5
    return weights

def extract_weights(traits, aggregation ,debug=False, use_natural_var=True):    
  var_N = np.var(traits,axis=0)
  mean_N = np.mean(traits, axis=0)
  var_O = np.var(aggregation,axis=0)
  mean_O = np.mean(aggregation,axis=0)
  cv_N = np.sqrt(var_N)/mean_N
  cv_O = np.sqrt(var_O)/mean_O
  natural_var = (1/np.max(cv_N))*cv_N
  observed_var = (1/np.max(cv_O))*cv_O
  
  if use_natural_var:
    weight = calculate_weight(natural_var,observed_var)
  else:
    weight = calculate_weight(1+(natural_var*0),observed_var)

  if debug:
    print("var_N",var_N)
    print("mean_N",mean_N)
    print("var_O",var_O)
    print("mean_O",mean_O)
    print("cv_N",cv_N)
    print("cv_O",cv_O)
    print("observed_var",observed_var)
    print("natural_var",natural_var)
    print("weight", weight)

  return weight

def get_weights(Q,Y, num_zeros=0, balance=False, normalize=False, use_natural_var=True, debug=False):

  defend = np.array([0.42899434, 0.23574219, 0.43544287, 0.27262153, 0.35855319,
        0.66831016, 0.91971063, 0.40302096, 0.43830869, 0.3748376 ,
        0.2396702 , 0.396162  , 0.71112441, 0.65870817, 0.36238951,
        0.13838584, 1.        , 0.13524566, 0.59085122, 0.43256051,
        0.53241872, 0.38262322, 0.19435433, 0.4410418 , 0.49800314,
        0.51829118, 0.17545001, 0.38318633, 0.64354782, 0.93134938,
        0.21438795, 0.        , 0.54286999, 0.67352307, 0.32245346,
        0.17931197, 0.51333105])

  forward = np.array([0.07511679, 0.09629094, 0.1318426 , 0.16192662, 1.        ,
        0.27951538, 0.43837041, 0.32260755, 0.43631347, 0.14078845,
        0.14117822, 0.22218995, 0.71190406, 0.29527659, 0.15439163,
        0.35573828, 0.37896582, 0.05864317, 0.59347301, 0.15286997,
        0.20169927, 0.33787653, 0.27498375, 0.24679628, 0.0924129 ,
        0.77589956, 0.24334671, 0.2294275 , 0.16841433, 0.24099061,
        0.16840206, 0.17367156, 0.        , 0.23989417, 0.15044812,
        0.24536376, 0.34687652])


  mid = np.array([0.34308533, 0.14161317, 0.37326579, 0.27585082, 0.3563625 ,
        0.2752294 , 0.80528806, 0.31595507, 0.47794581, 0.18584656,
        0.35213607, 0.20054489, 1.        , 0.08877895, 0.32165796,
        0.55976527, 0.47810338, 0.0278576 , 0.34095518, 0.2849252 ,
        0.39645807, 0.31186674, 0.23507877, 0.29159742, 0.21615502,
        0.12531428, 0.15870577, 0.19467947, 0.31975704, 0.30020646,
        0.23544479, 0.26155777, 0.24766322, 0.26570622, 0.41512273,
        0.06669918, 0.        ])


  gk = np.array([0.42037992, 0.39647297, 0.3918336 , 0.37364185, 0.41311582,
        0.48535614, 0.40716125, 0.34789179, 0.43371166, 0.42318088,
        0.30957407, 0.38575092, 0.41560188, 0.39689122, 0.41260252,
        0.39633157, 0.54019158, 0.40242688, 0.99630519, 0.38708318,
        0.38440996, 0.38662873, 0.32493371, 0.34677841, 0.3697066 ,
        0.27654126, 0.41720281, 0.40568391, 0.40031227, 0.3912252 ,
        0.38511077, 0.38083625, 1.        , 0.87849226, 0.        ,
        0.93959603, 0.96898746])
  weights = np.array([forward,mid,defend,gk])      
  # weights = []
  num_tasks = len(Y)
  # for i in range(num_tasks):
  #   weights.append(extract_weights(Q,Y[i],debug, use_natural_var))
  # weights = np.array(weights)
  
  if normalize:
    weights = weights / weights.sum(axis=1)[:, np.newaxis]

  if balance:
    weights = weights / weights.sum(axis=0)

  for m in range(num_tasks):
    idx = np.argsort(weights[m])
    k = num_zeros
    weights[m,idx[:k]] = 0

  return weights

def get_target_Q(dataset, nation, team_formation):
  team = dataset[dataset['nationality'] == nation]
  selection_team = []
  try:
    for i in range(len(POSITIONS)):
      temp = team[team['team_position']==POSITIONS[i]].sample(team_formation[i]).copy()
      selection_team.append(temp)
    final_team = pd.concat(selection_team)
  except:
    print(len(team))
    return None
  return final_team

def plot_tsme(dataset,Q,n_components=2,perplexity=25,init='pca',random_state=0):
  color = dataset['team_position'].copy()
  color[color == "Forward"] = "red"
  color[color == "Midfield"] = "blue"
  color[color == "GK"] = "lightpink"
  color[color == "Defense"] = "green"
  data_des = pd.DataFrame()
  data_des["color"] = color
  data_des["pos"] = dataset["team_position"]
  tsne = manifold.TSNE(n_components,perplexity,init,random_state)
  t0 = time()
  X_tsne = tsne.fit_transform(Q)
  print("t-SNE embedding (time %.2fs)" %(time() - t0))
  plot_figure = plot_embedding(X_tsne, data_des)
  return plot_figure

def plot_pca(dataset,Q,n_components=2):
  color = dataset['team_position'].copy()
  color[color == "Forward"] = "red"
  color[color == "Midfield"] = "blue"
  color[color == "GK"] = "lightpink"
  color[color == "Defense"] = "green"
  data_des = pd.DataFrame()
  data_des["color"] = color
  data_des["pos"] = dataset["team_position"]
  pca = decomposition.PCA(n_components)
  t0 = time()
  X_pca = pca.fit_transform(Q)
  print("PCA embedding of the digits (time %.2fs)" %(time() - t0))
  plot_figure = plot_embedding(X_pca,data_des)
  return plot_figure

def plot_embedding(data,data_des):
    color = data_des["color"]
    pos = data_des["pos"]
    m,n = data.shape
    if n == 3:
        task1 = go.Scatter3d(
            x=data[:,0],  # <-- Put your data instead
            y=data[:,1],  # <-- Put your data instead
            z=data[:,2],  # <-- Put your data instead
            mode='markers',
            text=pos,
            marker={
                'color':color,
                'size': 10,
                'opacity': 0.8,
            },)
    elif n == 2:
        task1 = go.Scatter(
                    x=data[:,0],  # <-- Put your data instead
                    y=data[:,1],  # <-- Put your data instead
                    mode='markers',
                    text=pos,
                    marker={
                        'color':color,
                        'size': 10,
                        'opacity': 0.8,
                    },)
    elif n == 1:
        task1 = go.Scatter(
                x=data[:,0],  # <-- Put your data instead
                y=np.zeros(m),  # <-- Put your data instead
                mode='markers',                    
                text=pos,
                marker={
                    'color':color,
                    'size': 10,
                    'opacity': 0.8,
                },)
    else:
        print("Dimensions received are ",n)
        return
    data = [task1]

    plot_figure = go.Figure(data=data)
    plot_figure.update_layout(width=700,
                        margin=dict(r=20, b=10, l=10, t=10))

    return plot_figure

