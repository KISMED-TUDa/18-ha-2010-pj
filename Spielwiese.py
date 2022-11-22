
import math
import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references
from scipy.fft import fft, fftfreq

samplingfrequenzy = 300;

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
sdnn_afib = np.array([])

N = ecg_leads[1].size;
T = 1.0/300;
x = np.linspace(0.0, N*T, N, endpoint=False);

y = ecg_leads[158];
yf = fft(y);
xf = fftfreq(N, T)[:N//2];
t = 2.0/N * np.abs(yf[0:N//2]);
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]));
plt.grid();
plt.show();
max_value = [];
i =1;
for i in range(0, 6000):
    y = ecg_leads[i];
    yf = fft(y);
    viel = np.abs(yf);
    d = max(viel);
    max_value.append(d); 
size = len(max_value);
q = 10;

#fig, ax = plt.subplots(figsize=(5, 5));
#ax.set_aspect(1);
#v = [i**2 for i in max_value];
#v = np.asarray(v);
#max_value = np.asarray(max_value);
#ax.scatter(max_value, v, color="r")
#plt.show();
#f= 1;