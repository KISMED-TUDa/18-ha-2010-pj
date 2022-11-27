# -*- coding: utf-8 -*-
"""
Diese Datei beinhaltet die Random Forrest Methode

requirements = scikit-learn (einfach mit pip install scikit-learn)

"""
__author__ = "Gruppe 112"

# import    Features

from wettbewerb import save_predictions
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from typing import List, Tuple

from features_112 import features
import features_112

def RandomForrest_112(ecg_leads: list[np.ndarray], ecg_labels : List[str], fs : int, ecg_names : List[str]) -> List[Tuple[str, str]] : 
    """
    Funktion wendet den RandomForrest an den Übergebenen Samples an und 
    gibt uns eine Liste von Tuples mit den Predicts raus. 
        
    Parameters
    ----------
    ecg_leads : List[np.ndarray]
        EKG Signale.
    ecg_labels : List[str]
        Gleiche Laenge wie ecg_leads. Werte: 'N','A','O','~'
    fs : int
        Sampling Frequenz.
    ecg_names : List[str]
        Name der geladenen Dateien    

    Returns
    -------
    Predictions : List[Tuple[str,str]]
    """

# Calculate the features
    features = features_112.features(ecg_leads,ecg_labels,fs,ecg_names)
    
    #features = features(ecg_leads,ecg_labels,fs,ecg_names)             --> das will er nicht checken auch mit methoden import

    # Array init
    
    labels = np.array([])                    # Array für labels mit 1(A) und 0(N)
    
    labels_string = np.array([])             # Array für labels mit A(1) und N(0)
    
    fail_label = np.array([])
    
    Prediction_array = np.array([])          # Array für Prediction
    
    Prediction_List = []                     # List für Prediction


# Change labels to 1 and 0; delete labels and related features with values != 0 or 1 
    
    for nr,y in enumerate(Prediction_array):

        if ecg_labels[nr] == 'N':                   # normal:   N = 0
            labels = np.append(labels,0)
            continue                                                        # continue damit der nicht ins else geht

        if ecg_labels[nr] == 'A':                   # Flimmern: A = 1
            labels = np.append(labels,1)
            continue

        #if ecg_labels[nr] != 'A' and ecg_labels[nr] != 'N':                # else wollte nicht klappen irgendwie
        
        else:                                       # ~ or O
            fail_label= np.append(fail_label, nr)

    
# delete every feature for the labels ~ and O
    
    features = np.delete(features, fail_label.astype(int), axis=0)


# Model und Algorithmus implementation

    model = RandomForestClassifier()

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)              # Kreuz validierung  

    Prediction_array = cross_val_predict( model, features, labels, cv)                  # prediction



# A und N zu 0 und 1 

    for nr,y in enumerate(ecg_labels):      # könnte man auch mit schleife ecg_labels[i] und 0<=i<ecg_labels.size()

        if ecg_labels[nr] == 0:                   # normal:   N = 0
            labels_string = np.append(labels_string,'N')
            continue                                                                                # continue damit der nicht ins else geht

        if ecg_labels[nr] == 1:                   # Flimmern: A = 1
            labels_string = np.append(labels_string,'A')
            continue

        #if ecg_labels[nr] != 'A' and ecg_labels[nr] != 'N':                                       # else wollte nicht klappen irgendwie
        else:
            labels_string = np.append(labels_string,0)           # noise und anderes : -1
       # pass

    
# jetzt Prediction_array erweitern mit ecg namen 

    #for i in Prediction_array:
    #    Prediction_array = Prediction_array[i][]



# Prediction array zu Prediction_List
    for i in Prediction_array:
        Prediction_List.append = Prediction_array[[i][i]]
        
        
    return Prediction_List    
    #List[tuple(str,str)]
    

