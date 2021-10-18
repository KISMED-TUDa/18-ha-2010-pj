# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:33:27 2021

@author: Maurice
"""

import os
import sys
import pandas as pd
from wettbewerb import save_predictions

if __name__=='__main__':
    
    # call: target folder [source folder1 source folder 2 ...]
    if len(sys.argv) <3 :
        raise Exception("No Predictions given")
    target_f = sys.argv[1]    
        
    predictions = list()    
    for i in range(2,len(sys.argv)):    
        predictions.append(pd.read_csv(sys.argv[i] + "PREDICTIONS.csv", header=None))
        assert (predictions[i-2].shape==predictions[0].shape),"Not equal prediction length!"
    n_predictors = len(predictions)
    ensemble_pred = list()
    for i in range(predictions[0].shape[0]):
        isA = 0
        for pre in predictions:
            isA += pre[1].values[i]=='A'
        if isA>=n_predictors/2:
            ensemble_pred.append((predictions[0][0].values[i],'A'))
        else:
            ensemble_pred.append((predictions[0][0].values[i],'N'))
    
    save_predictions(ensemble_pred,target_f)