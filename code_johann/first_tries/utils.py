import os
import numpy  as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

color_list = ["#" + i for i  in "264653-2a9d8f-e9c46a-f4a261-e76f51".split("-")]

def get_data(test_split = 0.2, seed = 42): 
    data_trace_high = pd.read_csv("/mnt/c/Users/johan/Desktop/CMT_project/data/traces_high.txt", sep = " ", header = None)
    data_trace_low  = pd.read_csv("/mnt/c/Users/johan/Desktop/CMT_project/data/traces_low.txt",  sep = " ", header = None)


    high_labels = np.ones(data_trace_high.shape[0])
    low_labels  = np.zeros(data_trace_low.shape[0])

    data        = pd.concat([data_trace_high, data_trace_low])
    labels      = np.concatenate([high_labels,     low_labels])

    return train_test_split(np.array(data), np.array(labels), test_size = test_split)

