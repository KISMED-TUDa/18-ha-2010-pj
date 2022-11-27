# -*- coding: utf-8 -*-
"""
Reads the training data and trains the RandomForrest Classifier

"""

__author__ = "Gruppe 112"

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from wettbewerb import load_references

from features_112 import features
import features_112

def RandomForrest_112(): 
    """
    Trains the RandomForrest classifier on the trainingset
        
    Parameters
    ----------
    None
    
    Returns
    -------
    Model : Trained Classifier
    """
########################### Load Trainings data ########################################################

    ecg_leads,ecg_labels,fs,ecg_names = load_references()


########################### Calculate the features ######################################################
    
    features = features_112.features(ecg_leads,ecg_labels,fs,ecg_names)
    
    #features = features(ecg_leads,ecg_labels,fs,ecg_names)             --> das will er nicht checken auch mit methoden import


########################### Array init  #################################################################
    
    labels = np.array([])                    # Array für labels mit 1(A) und 0(N)
        
    fail_label = np.array([])                # Array für labels mit ~ und O
    
    Prediction_array = np.array([])          # Array für Prediction


########################### Delete labels with values != 0 or 1 and corresponding features  ###############

    for nr,y in enumerate(ecg_labels):

        if ecg_labels[nr] == 'N':                   
            labels = np.append(labels,'N')            
            continue                                            # ohne continue geht er aus unerklärlichen gründen immer ins else

        if ecg_labels[nr] == 'A':                               # ""
            labels = np.append(labels,'A')
            continue

        else:
            fail_label= np.append(fail_label, nr)

    
########################### delete feature for the labels ~ and O    #########################################
    
    features = np.delete(features, fail_label.astype(int), axis=0)

########################### Model und Training ###############################################################

    model = RandomForestClassifier()
    
    model.fit(features,labels)

    return model