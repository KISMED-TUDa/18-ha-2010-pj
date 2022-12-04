# Test der Decision Trees

#import csv
#import scipy.io as sio

import matplotlib.pyplot as plt

#import pandas as pd

# evaluate random forest algorithm for classification
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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


features = features(ecg_leads,ecg_labels,fs,ecg_names); 

labels = np.array([])
fail_label = np.array([])


####        labels auf 0 und 1 

for nr,y in enumerate(ecg_labels):

    if ecg_labels[nr] == 'N':                   
        labels = np.append(labels,'N')            # scheinbar kann man N und A lassen     
        continue                                                                      # continue damit der nicht ins else geht

    if ecg_labels[nr] == 'A':                   # ""
        labels = np.append(labels,'A')
        continue

    #if ecg_labels[nr] != 'A' and ecg_labels[nr] != 'N':                                       # else wollte nicht klappen irgendwie
    else:
        fail_label= np.append(fail_label, nr)



features = np.delete(features, fail_label.astype(int), axis=0)

#############################################################   Split Train Test   ##########################

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=1)


#############################################################   Training    #################################

model = RandomForestClassifier()        # testen: max_features, warm_start 

#trained_model = model.fit(features,labels) 

model.fit(X_train,y_train)

#############################################################   Testing    #################################

Predictions = model.predict(X_test)

print(Predictions)
print(y_test)

#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)              # Kreuz validierung  

mean = model.score(X_test, y_test)

print('Accuracy: %.3f' % mean)

#n_f1 = cross_val_score(model, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')       # f1 fürs scoring

#n_accuracy = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')       # für uns 


# Printen für uns                                                    

#print('Accuracy: %.3f (%.3f)' % (np.mean(n_accuracy), np.std(n_accuracy)))                # Mittelwert und Standartdeviation

#print('Der F1 score: \n')

#print(n_f1)
