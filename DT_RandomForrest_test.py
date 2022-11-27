# Test der Decision Trees

#import csv
#import scipy.io as sio

import matplotlib.pyplot as plt

#import pandas as pd

# evaluate random forest algorithm for classification
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from wettbewerb import load_references
from features_112 import features


# 1.    Wir bekommen die EKG daten als Array
# 2.    Features aus den EKG daten extrahieren  -> Array reihe pro Feature wird geadded
# 3.    Algorithm gives us the prediction
# 4.    Compare with solution and Calculate Precision
 


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, Diagnose(A,N),
                                                      # Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)



# Change labels to 1 and 0
labels = np.array([])             # Ob Flimmern oder nicht                    --> Hier erwartet die klasse: 0,1 -> ja oder nein

noise=0
healthy=0
sick=0

fail_label = np.array([])
# Calculate the features

features = features(ecg_leads,ecg_labels,fs,ecg_names);                 

# Change labels to 1 and 0; delete labels and related features with values != 0 or 1 

for nr,y in enumerate(ecg_labels):

    if ecg_labels[nr] == 'N':                   # normal:   N = 0
        labels = np.append(labels,0)
        healthy+=1
        continue                                                                                # continue damit der nicht ins else geht

    if ecg_labels[nr] == 'A':                   # Flimmern: A = 1
        labels = np.append(labels,1)
        sick+=1
        continue

    #if ecg_labels[nr] != 'A' and ecg_labels[nr] != 'N':                                       # else wollte nicht klappen irgendwie
    else:
        fail_label= np.append(fail_label, nr)
        #features = np.delete(features, nr,0)
        #labels = np.append(labels,-1)           # noise und anderes : -1
        noise += 1

print("label shape:\n")
print(labels.shape)
print("NOISE:\n")
print(noise)
print("SICK:\n")
print(sick)
print("HEALTHY:\n")
print(healthy)

print("feature Shape:\n")           # rows:6000  -- column: 8
print(features.shape)

print("fail:\n")
print(fail_label.shape)


features = np.delete(features, fail_label.astype(int), axis=0)          # Delete every ~ or O in features




print("FEATURE SHAPE:\n")
print(features.shape)


# define the model
model = RandomForestClassifier()

# Kreuz validierung -- mehr trainingsdaten generieren
# k fold: https://datascientest.com/de/kreuzvalidierungsverfahren-definition-und-bedeutung-fur-machine-learning
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)        # Teil in 10 gruppen,            


                                                                                # VLLt testen der parameter verschiedenen
                                                                                # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve

# prediction

#Prediction = cross_val_predict( model, features,labels,cv)                                    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict



# Performance berechnung 

n_f1 = cross_val_score(model, features, labels, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')       # f1 fürs scoring

n_accuracy = cross_val_score(model, features, labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')       # für uns 


# Printen für uns                                                    

print('Accuracy: %.3f (%.3f)' % (np.mean(n_accuracy), np.std(n_accuracy)))                # Mittelwert und Standartdeviation

print('Der F1 score: \n')

print(n_f1)
