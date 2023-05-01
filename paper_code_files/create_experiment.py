from asyncore import write
import pandas as pd
import numpy as np
from utilities import *
import time
import os
import json
import argparse
from sklearn.model_selection import GroupKFold


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        os.makedirs(string)
        print("The new directory is created!")
        return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=int, 
        help="define the formation. Valid values are 4 or 11. Default is 11")
    parser.add_argument('-i', type=int, required=True, 
        help="define the number of iterations. Valid values are (0,100)")
    parser.add_argument('-path', type=dir_path, 
        help="define the folder the values should be saved into")
    args = parser.parse_args()

    formation = args.f
    iteration = args.i
    path = args.path

    if path is None:
        path = "experiment_"+str(time.strftime("%d_%b_%Y_%H_%M", time.gmtime()))
        path = dir_path(path)

    if iteration is None or iteration<1:
        iteration=10
    dataset = create_dataset("new_players_dataset.csv", True)

    if formation == 4:
        team_formation = np.array([1,1,1,1])
    else:
        team_formation = np.array([2,4,4,1])
    start_time = time.time()
    write_experiment_setup_to_file(dataset, team_formation,iteration,path)
    print("Time taken: ", time.time()-start_time)
