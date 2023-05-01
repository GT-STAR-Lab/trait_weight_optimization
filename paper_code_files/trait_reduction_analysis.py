# %%
import pandas as pd
import plotly.express as px
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import cvxpy as cp
from utilities import *
from time import time
# from scipy.stats import alexandergovern, f_oneway
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,GroupKFold
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)

# %%
def get_score_from_file(filename, return_raw=False):
    file = open(filename)
    data = json.load(file)
    scores = []
    for country in data.keys():
        for iteration in (data[country].keys()):
            value = data[country][iteration]['score']
            scores.append(value)
        
    total_possible = 11
    if not return_raw:
        print(scores)
        scores = np.array(scores)/total_possible
    return list(scores)

def get_time_from_file(filename):
    file = open(filename)
    data = json.load(file)
    times = []
    for country in data.keys():
        for iteration in (data[country].keys()):
            value = data[country][iteration]['time']
            times.append(value)
    return times

# %%
trait_1_time = get_time_from_file("one_iter_time_data/no_natural_score_1traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_1traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_1traits.json") 
trait_2_time = get_time_from_file("one_iter_time_data/no_natural_score_2traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_2traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_2traits.json") 
trait_3_time = get_time_from_file("one_iter_time_data/no_natural_score_3traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_3traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_3traits.json") 
trait_4_time = get_time_from_file("one_iter_time_data/no_natural_score_4traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_4traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_4traits.json") 
trait_5_time = get_time_from_file("one_iter_time_data/no_natural_score_5traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_5traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_5traits.json") 
trait_6_time = get_time_from_file("one_iter_time_data/no_natural_score_6traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_6traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_6traits.json") 
trait_7_time = get_time_from_file("one_iter_time_data/no_natural_score_7traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_7traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_7traits.json") 
trait_8_time = get_time_from_file("one_iter_time_data/no_natural_score_8traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_8traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_8traits.json") 
trait_9_time = get_time_from_file("one_iter_time_data/no_natural_score_9traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_9traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_9traits.json") 
trait_10_time = get_time_from_file("one_iter_time_data/no_natural_score_10traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_10traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_10traits.json") 
trait_11_time = get_time_from_file("one_iter_time_data/no_natural_score_11traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_11traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_11traits.json") 
trait_12_time = get_time_from_file("one_iter_time_data/no_natural_score_12traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_12traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_12traits.json") 
trait_13_time = get_time_from_file("one_iter_time_data/no_natural_score_13traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_13traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_13traits.json") 
trait_14_time = get_time_from_file("one_iter_time_data/no_natural_score_14traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_14traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_14traits.json") 
trait_15_time = get_time_from_file("one_iter_time_data/no_natural_score_15traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_15traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_15traits.json") 
trait_16_time = get_time_from_file("one_iter_time_data/no_natural_score_16traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_16traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_16traits.json") 
trait_17_time = get_time_from_file("one_iter_time_data/no_natural_score_17traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_17traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_17traits.json") 
trait_18_time = get_time_from_file("one_iter_time_data/no_natural_score_18traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_18traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_18traits.json") 
trait_19_time = get_time_from_file("one_iter_time_data/no_natural_score_19traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_19traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_19traits.json") 
trait_20_time = get_time_from_file("one_iter_time_data/no_natural_score_20traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_20traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_20traits.json") 
trait_21_time = get_time_from_file("one_iter_time_data/no_natural_score_21traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_21traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_21traits.json") 
trait_22_time = get_time_from_file("one_iter_time_data/no_natural_score_22traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_22traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_22traits.json") 
trait_23_time = get_time_from_file("one_iter_time_data/no_natural_score_23traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_23traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_23traits.json") 
trait_24_time = get_time_from_file("one_iter_time_data/no_natural_score_24traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_24traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_24traits.json") 
trait_25_time = get_time_from_file("one_iter_time_data/no_natural_score_25traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_25traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_25traits.json") 
trait_26_time = get_time_from_file("one_iter_time_data/no_natural_score_26traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_26traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_26traits.json") 
trait_27_time = get_time_from_file("one_iter_time_data/no_natural_score_27traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_27traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_27traits.json") 
trait_28_time = get_time_from_file("one_iter_time_data/no_natural_score_28traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_28traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_28traits.json") 
trait_29_time = get_time_from_file("one_iter_time_data/no_natural_score_29traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_29traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_29traits.json") 
trait_30_time = get_time_from_file("one_iter_time_data/no_natural_score_30traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_30traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_30traits.json") 
trait_31_time = get_time_from_file("one_iter_time_data/no_natural_score_31traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_31traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_31traits.json") 
trait_32_time = get_time_from_file("one_iter_time_data/no_natural_score_32traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_32traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_32traits.json") 
trait_33_time = get_time_from_file("one_iter_time_data/no_natural_score_33traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_33traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_33traits.json") 
trait_34_time = get_time_from_file("one_iter_time_data/no_natural_score_34traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_34traits.json")+get_time_from_file("one_iter_time_data/3_no_natural_score_34traits.json") 
trait_35_time = get_time_from_file("one_iter_time_data/no_natural_score_35traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_35traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_35traits.json")
trait_36_time = get_time_from_file("one_iter_time_data/no_natural_score_36traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_36traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_36traits.json")
trait_37_time = get_time_from_file("one_iter_time_data/no_natural_score_37traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_37traits.json")+get_time_from_file("one_iter_time_data/2_no_natural_score_37traits.json")
base_time = get_time_from_file("one_iter_time_data/base.json")+get_time_from_file("one_iter_time_data/base_2.json")+get_time_from_file("one_iter_time_data/base_3.json") 


# %%
trait_1score = get_score_from_file("one_iter_time_data/no_natural_score_1traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_1traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_1traits.json") 
trait_2score = get_score_from_file("one_iter_time_data/no_natural_score_2traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_2traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_2traits.json") 
trait_3score = get_score_from_file("one_iter_time_data/no_natural_score_3traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_3traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_3traits.json") 
trait_4score = get_score_from_file("one_iter_time_data/no_natural_score_4traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_4traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_4traits.json") 
trait_5score = get_score_from_file("one_iter_time_data/no_natural_score_5traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_5traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_5traits.json") 
trait_6score = get_score_from_file("one_iter_time_data/no_natural_score_6traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_6traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_6traits.json") 
trait_7score = get_score_from_file("one_iter_time_data/no_natural_score_7traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_7traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_7traits.json") 
trait_8score = get_score_from_file("one_iter_time_data/no_natural_score_8traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_8traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_8traits.json") 
trait_9score = get_score_from_file("one_iter_time_data/no_natural_score_9traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_9traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_9traits.json") 
trait_10score = get_score_from_file("one_iter_time_data/no_natural_score_10traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_10traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_10traits.json") 
trait_11score = get_score_from_file("one_iter_time_data/no_natural_score_11traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_11traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_11traits.json") 
trait_12score = get_score_from_file("one_iter_time_data/no_natural_score_12traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_12traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_12traits.json") 
trait_13score = get_score_from_file("one_iter_time_data/no_natural_score_13traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_13traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_13traits.json") 
trait_14score = get_score_from_file("one_iter_time_data/no_natural_score_14traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_14traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_14traits.json") 
trait_15score = get_score_from_file("one_iter_time_data/no_natural_score_15traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_15traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_15traits.json") 
trait_16score = get_score_from_file("one_iter_time_data/no_natural_score_16traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_16traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_16traits.json") 
trait_17score = get_score_from_file("one_iter_time_data/no_natural_score_17traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_17traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_17traits.json") 
trait_18score = get_score_from_file("one_iter_time_data/no_natural_score_18traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_18traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_18traits.json") 
trait_19score = get_score_from_file("one_iter_time_data/no_natural_score_19traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_19traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_19traits.json") 
trait_20score = get_score_from_file("one_iter_time_data/no_natural_score_20traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_20traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_20traits.json") 
trait_21score = get_score_from_file("one_iter_time_data/no_natural_score_21traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_21traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_21traits.json") 
trait_22score = get_score_from_file("one_iter_time_data/no_natural_score_22traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_22traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_22traits.json") 
trait_23score = get_score_from_file("one_iter_time_data/no_natural_score_23traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_23traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_23traits.json") 
trait_24score = get_score_from_file("one_iter_time_data/no_natural_score_24traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_24traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_24traits.json") 
trait_25score = get_score_from_file("one_iter_time_data/no_natural_score_25traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_25traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_25traits.json") 
trait_26score = get_score_from_file("one_iter_time_data/no_natural_score_26traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_26traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_26traits.json") 
trait_27score = get_score_from_file("one_iter_time_data/no_natural_score_27traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_27traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_27traits.json") 
trait_28score = get_score_from_file("one_iter_time_data/no_natural_score_28traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_28traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_28traits.json") 
trait_29score = get_score_from_file("one_iter_time_data/no_natural_score_29traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_29traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_29traits.json") 
trait_30score = get_score_from_file("one_iter_time_data/no_natural_score_30traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_30traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_30traits.json") 
trait_31score = get_score_from_file("one_iter_time_data/no_natural_score_31traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_31traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_31traits.json") 
trait_32score = get_score_from_file("one_iter_time_data/no_natural_score_32traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_32traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_32traits.json") 
trait_33score = get_score_from_file("one_iter_time_data/no_natural_score_33traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_33traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_33traits.json") 
trait_34score = get_score_from_file("one_iter_time_data/no_natural_score_34traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_34traits.json")+get_score_from_file("one_iter_time_data/3_no_natural_score_34traits.json") 
trait_35score = get_score_from_file("one_iter_time_data/no_natural_score_35traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_35traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_35traits.json")
trait_36score = get_score_from_file("one_iter_time_data/no_natural_score_36traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_36traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_36traits.json")
trait_37score = get_score_from_file("one_iter_time_data/no_natural_score_37traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_37traits.json")+get_score_from_file("one_iter_time_data/2_no_natural_score_37traits.json")
base_score = get_score_from_file("one_iter_time_data/base.json")+get_score_from_file("one_iter_time_data/base_2.json")+get_score_from_file("one_iter_time_data/base_3.json") 


# %%
len(trait_1_time), len(trait_1score)

# %%
# for i in range(37):
#     print("trait_{price}score = get_score_from_file(\"one_iter_time_data/no_natural_score_{price}traits.json\")+get_score_from_file(\"one_iter_time_data/2_no_natural_score_{price}traits.json\")+get_score_from_file(\"one_iter_time_data/3_no_natural_score_{price}traits.json\") ".format(price = i+1))

# %%
import plotly.express as px

time_df = pd.DataFrame({
'trait_1_time':trait_1_time,
'trait_2_time':trait_2_time,
'trait_3_time':trait_3_time,
'trait_4_time':trait_4_time,
'trait_5_time':trait_5_time,
'trait_6_time':trait_6_time,
'trait_7_time':trait_7_time,
'trait_8_time':trait_8_time,
'trait_9_time':trait_9_time,
'trait_10_time':trait_10_time,
'trait_11_time':trait_11_time,
'trait_12_time':trait_12_time,
'trait_13_time':trait_13_time,
'trait_14_time':trait_14_time,
'trait_15_time':trait_15_time,
'trait_16_time':trait_16_time,
'trait_17_time':trait_17_time,
'trait_18_time':trait_18_time,
'trait_19_time':trait_19_time,
'trait_20_time':trait_20_time,
'trait_21_time':trait_21_time,
'trait_22_time':trait_22_time,
'trait_23_time':trait_23_time,
'trait_24_time':trait_24_time,
'trait_25_time':trait_25_time,
'trait_26_time':trait_26_time,
'trait_27_time':trait_27_time,
'trait_28_time':trait_28_time,
'trait_29_time':trait_29_time,
'trait_30_time':trait_30_time,
'trait_31_time':trait_31_time,
'trait_32_time':trait_32_time,
'trait_33_time':trait_33_time,
'trait_34_time':trait_34_time,
'trait_35_time':trait_35_time,
'trait_36_time':trait_36_time,
'trait_37_time':trait_37_time,
'base_time':base_time
})

score_df = pd.DataFrame({'trait_1score':trait_1score,
'trait_2score':trait_2score,
'trait_3score':trait_3score,
'trait_4score':trait_4score,
'trait_5score':trait_5score,
'trait_6score':trait_6score,
'trait_7score':trait_7score,
'trait_8score':trait_8score,
'trait_9score':trait_9score,
'trait_10score':trait_10score,
'trait_11score':trait_11score,
'trait_12score':trait_12score,
'trait_13score':trait_13score,
'trait_14score':trait_14score,
'trait_15score':trait_15score,
'trait_16score':trait_16score,
'trait_17score':trait_17score,
'trait_18score':trait_18score,
'trait_19score':trait_19score,
'trait_20score':trait_20score,
'trait_21score':trait_21score,
'trait_22score':trait_22score,
'trait_23score':trait_23score,
'trait_24score':trait_24score,
'trait_25score':trait_25score,
'trait_26score':trait_26score,
'trait_27score':trait_27score,
'trait_28score':trait_28score,
'trait_29score':trait_29score,
'trait_30score':trait_30score,
'trait_31score':trait_31score,
'trait_32score':trait_32score,
'trait_33score':trait_33score,
'trait_34score':trait_34score,
'trait_35score':trait_35score,
'trait_36score':trait_36score,
'trait_37score':trait_37score,
'base_score':base_score

})

# %%


# %%
fig = px.box(score_df,x=score_df.columns)
fig.update_layout( xaxis_title="Allocation success percentage", yaxis_title="Algorithm used to allocation")
fig.show()

# %%
fig = px.violin(time_df, x=time_df.columns)
fig.update_layout( xaxis_title="Allocation success percentage", yaxis_title="Algorithm used to allocation")
fig.show()

#
# %%
plt.rcParams["figure.figsize"] = (15,14)

# %%
from scipy.interpolate import make_interp_spline

# Create a plot
x_val = np.array(range(0,39))

time_means = np.mean(time_df.values,axis=0)
time_means = np.insert(time_means, 0, 0)
time_means = time_means/time_means[-1]

time_spline = make_interp_spline(x_val, time_means)
# Create a plot
score_mean = np.mean(score_df.values,axis=0)/np.mean(score_df["base_score"].values,axis=0)
score_lo_std = (np.mean(score_df.values,axis=0) - np.std(score_df.values,axis=0))/np.mean(score_df["base_score"].values,axis=0)
score_hi_std = (np.mean(score_df.values,axis=0) + np.std(score_df.values,axis=0))/np.mean(score_df["base_score"].values,axis=0)
score_mean = np.insert(score_mean, 0, 0)
score_lo_std = np.insert(score_lo_std, 0, 0)
score_hi_std = np.insert(score_hi_std, 0, 0)


score_lo_std[1] -= 0.1
score_spline = make_interp_spline(x_val, score_mean)
score_lo_spline = make_interp_spline(x_val, score_lo_std)
score_hi_spline = make_interp_spline(x_val, score_hi_std)

X_score = np.linspace(x_val.min(), x_val.max(), 500)
X_time = np.linspace(x_val.min(), x_val.max(), 70)

time_Y = time_spline(X_time)
# score_Y = score_spline(X_score)
# score_lo_Y = score_lo_spline(X_score)
# score_hi_Y = score_hi_spline(X_score)

fig, ax = plt.subplots()
p1, = ax.plot(score_mean, linewidth=5, color="C6", marker="x", label="Score")
ax.set_ylabel("Normalized average percent of succesfull allocations", color="black",fontsize=25)

ax2 = ax.twinx()
ax.fill_between(x_val,score_lo_std,score_mean,color="C6",alpha=0.2)
ax.fill_between(x_val,score_hi_std,score_mean, color="C6",alpha=0.2)
time_Y[38] = 1
p2, = ax2.plot(time_Y,linewidth=5,color="C2",marker="o", label="Time")
ax2.set_ylabel("Normalized average time in seconds", color="black",fontsize=25)
ax.set_xlabel("Number of Traits", color="Black",fontsize=32)

ax.set_ylim(0, 1.3)
ax2.set_ylim(0, 1.3)
ax.set_xlim(0, 38)
ax.legend(handles=[p1, p2], loc ='upper left',fontsize=18)
# plt.xticks(range(38))
plt.show()
