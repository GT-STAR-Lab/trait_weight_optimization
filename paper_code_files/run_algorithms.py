from asyncore import write
from locale import normalize
from pickle import FALSE
import string
from tokenize import String
import pandas as pd
import numpy as np
from utilities import *
import time
import os
import json
import argparse
from sklearn.model_selection import GroupKFold

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, help="number of traits")
    parser.add_argument('-p', type=str, help="input path")
    parser.add_argument('-o', type=str, help="output path")
    parser.add_argument('-n', help="normalized", type=str2bool, nargs='?',
                        const=True, default=True,)
    parser.add_argument('-v', help="natural variation", type=str2bool, nargs='?',
                        const=True, default=True,)

    args = parser.parse_args()

    requested_traits = args.t

    path = args.p
    if path is None:
        path = "one_iter_data"
    out_folder = args.o
    if out_folder is None:
        out_folder = "new_results"
    dataset = create_dataset("new_players_dataset.csv", True)

    team_formation = np.array([2,4,4,1])

    info, teams = load_experiment_setup_to_file(dataset,path, team_formation)
    print("Loading completed from", path)
    
    num_traits = info[0]['Q'].shape[1]
    if requested_traits > num_traits:
        requested_traits = num_traits
    
    is_normalize = args.n
    use_natural_var = args.v

    num_zeros = num_traits - requested_traits
    print("Requested traits:", requested_traits, "number of zeros:", num_zeros)
    start_time = time.time()
    score_traits = run_experiment(info,teams, team_formation, num_zeros=num_zeros, normalize=is_normalize, use_natural_var=use_natural_var)
    print("time taken:",time.time()-start_time)
    print("Experiment done")

    filename = str.format(out_folder+'/'+str(is_normalize)+'_norm_'+str(use_natural_var)+'_natural_score_{count:d}traits_reg_weight.json',count = requested_traits)

    f = open(filename, 'w')
    json_string = json.dumps(score_traits,indent=2)
    f.write(json_string)
    f.close()
    print("file written")
