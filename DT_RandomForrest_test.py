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

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, Diagnose(A,N),
                                                      # Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)



################################################################## Array + Debugging stuff init

labels = np.array([])               # Array für labels mit 1(A) und 0(N)
fail_label = np.array([])           # Array für labels mit ~ und O

noise=0
healthy=0
sick=0


################################################################## Calculate the features

features = features(ecg_leads,ecg_labels,fs,ecg_names);                 


################################################################## Change labels to 1 and 0

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


################################################################## delete labels and related features with values != 0 or 1 

features = np.delete(features, fail_label.astype(int), axis=0)          # Delete every ~ or O in features


################################################################## Debugging stuff 

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


###################################################################  Trainings und Test Satz Split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=1)

##################################################################  Modell und Training 

model = RandomForestClassifier()

model.fit(X_train,y_train)

##################################################################  Prediction

Predictions = model.predict(X_test)
print("PREDICTION:\n")
print(Predictions)
print(y_test)


##################################################################  Performance berechnung 

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)        # Teil in 10 gruppen,            


n_f1 = cross_val_score(model, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')       # f1 fürs scoring

n_accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')       # für uns 


# Printen für uns                                                    

print('Accuracy: %.3f (%.3f)' % (np.mean(n_accuracy), np.std(n_accuracy)))                # Mittelwert und Standartdeviation

print('Der F1 score: \n')

print(n_f1)
