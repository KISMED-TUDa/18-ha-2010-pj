# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""



import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from scipy.fft import fft, fftfreq
from wettbewerb import load_references

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

# Feature-List
sdnn_normal = np.array([])                                # Initialisierung normal ("N") SDNN
sdnn_afib = np.array([])                                  # Initialisierung afib ("A") SDNN
peak_diff_mean = np.array([])                             # Initialisierung mean R-Peak difference
peak_diff_median = np.array([])                           # Initialisierung median R-Peak difference
peaks_per_measure = np.array([])                          # Initialisierung R-Peaks per meassurement
peaks_per_lowPass = np.array([])                          # Initialisierung R-Peaks per meassurement
max_amplitude = np.array([])
relativ_lowPass = np.array([])
relativ_highPass = np.array([])
relativ_bandPass = np.array([])

### FFT Initialisierung
N = ecg_leads[1].size;                                  # Anzahl der Messungen (9000 in 30s)
T = 1.0/300.0;                                          # Kalibrierung auf Sampel-Frequenz
t = np.linspace(0.0, N*T, N, endpoint=False);           # Initialisierung des Zeitbereiches (für jede Messung gleich, daher nur einemal berechnet)
xf = fftfreq(N, T)[:N//2];                              # Initialisierung des Frequenzbereiches
max_value = [];                 

for idx, ecg_lead in enumerate(ecg_leads):
    ### Zeitbereich
    r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
    peak_to_peak_diff = (np.diff(r_peaks))              # Abstände der R-Spitzen
    sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden

    ### Frequenzbereich    
    y = ecg_lead                                        # Laden des Messung
    if len(y)<4499 :
        d = 4500-len(y)
        for i in range(0,d):
            y = np.append(y, 0)
    yf = fft(y)                                         # Komplexes Spektrum
    r_yf = 2.0/N * np.abs(yf[0:N//2])                   # Reelles Spektrum
    normier_faktor = 1/(np.sum(r_yf))                   # Inverses Integral über Frequenzbereich
    

    ### LowPass Filter
    yf_lowPass = np.array([]);
    for i in range(0,450):
       yf_lowPass = np.append(yf_lowPass, r_yf[i])
    #normier_faktor_lowPass = 1/(np.sum(yf_lowPass))                   # Inverses Integral über Frequenzbereich

    ### BandPass Filter
    yf_bandPass = np.array([]);
    for i in range(451,3500):
        yf_bandPass = np.append(yf_bandPass, r_yf[i])
    #normier_faktor_lowPass = 1/(np.sum(yf_lowPass))                   # Inverses Integral über Frequenzbereich

    ### HighPass Filter
    yf_highPass = np.array([]);
    for i in range(3501,3999):
        yf_highPass = np.append(yf_highPass, r_yf[i])
    #normier_faktor_lowPass = 1/(np.sum(yf_lowPass))                   # Inverses Integral über Frequenzbereich
    
    ### Relatives Gewicht der Unter-, Mittel- und Oberfreqeunzen
    relativ_lowPass = np.append(relativ_lowPass, np.sum(yf_lowPass)/normier_faktor)
    relativ_bandPass = np.append(relativ_bandPass, np.sum(yf_bandPass)/normier_faktor)
    relativ_highPass = np.append(relativ_highPass, np.sum(yf_highPass)/normier_faktor)
    max_amplitude = np.append(max_amplitude, max(r_yf))
    peaks_per_measure = np.append(peaks_per_measure, len(r_peaks))
    peak_diff_mean = np.append(peak_diff_mean, np.mean(peak_to_peak_diff))
    peak_diff_median = np.append(peak_diff_median, np.median(peak_to_peak_diff))

### Label abhängige Features?!?
    if ecg_labels[idx]=='N':
      sdnn_normal = np.append(sdnn_normal,sdnn)         # Zuordnung zu "Normal"
    if ecg_labels[idx]=='A':
      sdnn_afib = np.append(sdnn_afib,sdnn)             # Zuordnung zu "Vorhofflimmern"
    if (idx % 100)==0:
      print(str(idx) + "\t EKG Signale wurden verarbeitet.")


for i in range(0, 6000):
    y = ecg_leads[i];
    yf = fft(y);
    viel = np.abs(yf);
    d = max(viel);
    max_value.append(d); 
size = len(max_value);


fig, axs = plt.subplots(2,1, constrained_layout=True)
axs[0].hist(sdnn_normal,2000)
axs[0].set_xlim([0, 300])
axs[0].set_title("Normal")
axs[0].set_xlabel("SDNN (ms)")
axs[0].set_ylabel("Anzahl")
axs[1].hist(sdnn_afib,300)
axs[1].set_xlim([0, 300])
axs[1].set_title("Vorhofflimmern")
axs[1].set_xlabel("SDNN (ms)")
axs[1].set_ylabel("Anzahl")
plt.show()

sdnn_total = np.append(sdnn_normal,sdnn_afib) # Kombination der beiden SDNN-Listen
p05 = np.nanpercentile(sdnn_total,5)          # untere Schwelle
p95 = np.nanpercentile(sdnn_total,95)         # obere Schwelle
thresholds = np.linspace(p05, p95, num=20)    # Liste aller möglichen Schwellwerte
F1 = np.array([])
for th in thresholds:
  TP = np.sum(sdnn_afib>=th)                  # Richtig Positiv
  TN = np.sum(sdnn_normal<th)                 # Richtig Negativ
  FP = np.sum(sdnn_normal>=th)                # Falsch Positiv
  FN = np.sum(sdnn_afib<th)                   # Falsch Negativ
  F1 = np.append(F1, TP / (TP + 1/2*(FP+FN))) # Berechnung des F1-Scores

th_opt=thresholds[np.argmax(F1)]              # Bestimmung des Schwellwertes mit dem höchsten F1-Score

if os.path.exists("model.npy"):
    os.remove("model.npy")
with open('model.npy', 'wb') as f:
    np.save(f, th_opt)

fig, ax = plt.subplots()
ax.plot(thresholds,F1)
ax.plot(th_opt,F1[np.argmax(F1)],'xr')
ax.set_title("Schwellwert")
ax.set_xlabel("SDNN (ms)")
ax.set_ylabel("F1")
plt.show()

fig, axs = plt.subplots(2,1, constrained_layout=True)
axs[0].hist(sdnn_normal,2000)
axs[0].set_xlim([0, 300])
tmp = axs[0].get_ylim()
axs[0].plot([th_opt,th_opt],[0,10000])
axs[0].set_ylim(tmp)
axs[0].set_title("Normal")
axs[0].set_xlabel("SDNN (ms)")
axs[0].set_ylabel("Anzahl")
axs[1].hist(sdnn_afib,300)
axs[1].set_xlim([0, 300])
tmp = axs[1].get_ylim()
axs[1].plot([th_opt,th_opt],[0,10000])
axs[1].set_ylim(tmp)
axs[1].set_title("Vorhofflimmern")
axs[1].set_xlabel("SDNN (ms)")
axs[1].set_ylabel("Anzahl")
plt.show()