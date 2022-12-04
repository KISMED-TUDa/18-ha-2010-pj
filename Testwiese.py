# Test der Decision Trees

#import csv
#import scipy.io as sio

import matplotlib.pyplot as plt

#import pandas as pd

# evaluate random forest algorithm for classification
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from wettbewerb import load_references
from features_112 import features


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
from train import RandomForrest_112
import features_112


ecg_leads,ecg_labels,fs,ecg_names = load_references()

features = features_112.features(ecg_leads,fs,ecg_names)

########################### Use features and trained model to predict ########################################################

Predictions_array = RandomForrest_112().predict(features)

predictions = []
for nr,y in enumerate(Predictions_array):           # ecg_names = List ; Pred = Array
    
    predictions.append((ecg_names[nr],y))
    
print(predictions)