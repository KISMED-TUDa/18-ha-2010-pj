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
import math

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references()     # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

# Label-List
labels = np.array([])                                     # Initialisierung Array für die Labels.


# Feature-List
# alt
sdnn_normal = np.array([])                                # Initialisierung normal ("N") SDNN.
# alt
sdnn_afib = np.array([])                                  # Initialisierung afib ("A") SDNN.
# neu
sdnn = np.array([])
peak_diff_mean = np.array([])                             # Initialisierung Mittelwert des R-Spitzen Abstand.
peak_diff_median = np.array([])                           # Initialisierung Median des R-Spitzen Abstand.
peaks_per_measure = np.array([])                          # Initialisierung Anzahl der R-Spitzen.
peaks_per_lowPass = np.array([])                          # Initialisierung R-Spitzen im Nierderfrequenzbereich.
max_amplitude = np.array([])                              # Initialisierung Maximaler Auschlage des Spannungspegels. 
relativ_lowPass = np.array([])                            # Initialisierung Relativer Anteil des Niederfrequenzbandes an dem Gesamtspektrum.
relativ_highPass = np.array([])                           # Initialisierung Relativer Anteil des Mittelfrequenzbandes an dem Gesamtspektrum.
relativ_bandPass = np.array([])                           # Initialisierung Relativer Anteil des Hochfrequenzbandes an dem Gesamtspektrum.
rmssd = np.array([])                                      # Initialisierung des RMSSD Wertes

### FFT Initialisierung
N = ecg_leads[1].size                                     # Anzahl der Messungen (9000 in 30s, für jede Messung gleich, daher nur einemal berechnet).
fs = 300                                                  # Gegebene Abtastfrequenz des Messung.
T = 1.0/300.0                                             # Kalibrierung auf Sampel-Frequenz/Abtastungsrate (300Hz).
fd = fs/N                                                 # Frequenzauflösung des Spektrumes der Messung. !Nyquistkriterium: Es können höchstens bis 150Hz aussagekräftige Informationen gewonnen werden!
t = np.linspace(0.0, N*T, N, endpoint=False);             # Initialisierung des Zeitbereiches (für jede Messung gleich, daher nur einemal berechnet).
xf = fftfreq(N, T)[:N//2];                                # Initialisierung des Frequenzbereiches (für jede Messung gleich, daher nur einemal berechnet).


### Wenn Testlauf, dann können in range(102,6000) Messungen gelöscht werden, welche dann nicht mehr verarbietet werden.
ecg_leads = np.delete(ecg_leads, range(102,6000))


### Datenverarbeitung für jede Messung. Die Ergebnisse werden in den Arrays der Feature-List gespeichert.
for idx, ecg_lead in enumerate(ecg_leads):

    ### Zeitbereich
    r_peaks = detectors.hamilton_detector(ecg_lead)       # Detektion der QRS-Komplexe.
    peak_to_peak_diff = (np.diff(r_peaks))                # Abstände der R-Spitzen.
    sdnn_value = np.std(np.diff(r_peaks)/fs*1000)         # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden.
    

    ### Frequenzbereich    
    y = ecg_lead                                          # Laden des Messung
    if len(y)<4499 :                                      # Bei weniger Messungen (<9000) werden "0" an den Array gehängt.
        d = 4500-len(y)
        for i in range(0,d):
            y = np.append(y, 0)
    yf = fft(y)                                           # Berechnung des komplexen Spektrums.
    r_yf = 2.0/N * np.abs(yf[0:N//2])                     # Umwandlung in ein reelles Spektrum.
    normier_faktor = 1/(np.sum(r_yf))                     # Inverses Integral über Frequenzbereich
    

    ### LowPass Filter
    yf_lowPass = np.array([]);                            # Tiefpassfilter von Frequenz (0-450)*fd, dass entspricht (0-15)Hz.
    for i in range(0,450):
       yf_lowPass = np.append(yf_lowPass, r_yf[i])

    ### BandPass Filter
    yf_bandPass = np.array([]);                           # Bandpassfilter von Frequenz (451-3500)*fd, dass entspricht (15-116)Hz.
    for i in range(451,3500):
        yf_bandPass = np.append(yf_bandPass, r_yf[i])

    ### HighPass Filter                                   # Hochpassfilter von Frequenz (3501-3999)*fd, dass entspricht (116-133)Hz.
    yf_highPass = np.array([]);
    for i in range(3501,3999):
        yf_highPass = np.append(yf_highPass, r_yf[i])
    
    ### Features:       Relatives Gewicht der Unter-, Mittel- und Oberfreqeunzen.
    relativ_lowPass = np.append(relativ_lowPass, np.sum(yf_lowPass)/normier_faktor)
    relativ_bandPass = np.append(relativ_bandPass, np.sum(yf_bandPass)/normier_faktor)
    relativ_highPass = np.append(relativ_highPass, np.sum(yf_highPass)/normier_faktor)

    ### Feature:       Maximaler Ausschlag/Amplitude einer Messung.
    max_amplitude = np.append(max_amplitude, max(r_yf))

    ### Features:       R-Spitzen Abstand und Anzahl einer Messung.
    peaks_per_measure = np.append(peaks_per_measure, len(r_peaks))
    peak_diff_mean = np.append(peak_diff_mean, np.mean(peak_to_peak_diff))
    peak_diff_median = np.append(peak_diff_median, np.median(peak_to_peak_diff))

    ### Feature:        Anzahl an Spektrum-Spitzen im Niederfrequenzband.
    max_peak_sp = max(r_yf)                               # Ermittlung der höchsten Spitze.
    peaks_low = np.array([])                    
    for i in range(0, 4500):                   # Alle Spitzen übernehmen welche 80% der  höchsten Spitze erreichen.
        if r_yf[i] > 0.8*max_peak_sp:
            peaks_low = np.append(peaks_low, r_yf[i])
    peaks_per_lowPass = np.append(peaks_per_lowPass, peaks_low.size)  # Ermittlung der Anzahl der Spitzen mit mindesten 80% der maximal Spitze.
   
    ### Feature:        RMSSD
    n = peak_to_peak_diff.size                 # Anzahl an R-Spitzen-Abständen
    sum = 0.0
    for i in range(0, n-2):                    # Berechnung des RMSSD-Wertes
        sum += (peak_to_peak_diff[i + 1] - peak_to_peak_diff[i])**2
    rmssd = np.append(rmssd, math.sqrt(1/(n-1)*sum))

    ### Label-Erkennung und Zuweisung zu den Features.
    if ecg_labels[idx]=='N':
      # alt    # sdnn_normal = np.append(sdnn_normal,sdnn_value)         # Zuordnung zu "Normal"
      labels = np.append(labels, 'N')
    if ecg_labels[idx]=='A':
      # alt    # sdnn_afib = np.append(sdnn_afib,sdnn_value)             # Zuordnung zu "Vorhofflimmern"
      labels = np.append(labels, 'A')
    if ecg_labels[idx]=='O':
          labels = np.append(labels, 'O')
    if ecg_labels[idx]=='~':
          labels = np.append(labels, '~')
    if (idx % 100)==0:
      print(str(idx) + "\t EKG Signale wurden verarbeitet.")


## Erstellen der Feature-Matrix inklusive der Labels.
features = np.array([[labels], [peak_diff_mean], [peak_diff_median], [peaks_per_measure], [peaks_per_lowPass], [max_amplitude], [relativ_lowPass], [relativ_highPass], [relativ_bandPass], [rmssd]])


## Erstellen eines Diagrammes.
#fig, axs = plt.subplots(2,1, constrained_layout=True)
#axs[0].hist(sdnn_normal,2000)
#axs[0].set_xlim([0, 300])
#axs[0].set_title("Normal")
#axs[0].set_xlabel("SDNN (ms)")
#axs[0].set_ylabel("Anzahl")
#axs[1].hist(sdnn_afib,300)
#axs[1].set_xlim([0, 300])
#axs[1].set_title("Vorhofflimmern")
#axs[1].set_xlabel("SDNN (ms)")
#axs[1].set_ylabel("Anzahl")
#plt.show()

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

#fig, ax = plt.subplots()
#ax.plot(thresholds,F1)
#ax.plot(th_opt,F1[np.argmax(F1)],'xr')
#ax.set_title("Schwellwert")
#ax.set_xlabel("SDNN (ms)")
#ax.set_ylabel("F1")
#plt.show()

#fig, axs = plt.subplots(2,1, constrained_layout=True)
#axs[0].hist(sdnn_normal,2000)
#axs[0].set_xlim([0, 300])
#tmp = axs[0].get_ylim()
#axs[0].plot([th_opt,th_opt],[0,10000])
#axs[0].set_ylim(tmp)
#axs[0].set_title("Normal")
#axs[0].set_xlabel("SDNN (ms)")
#axs[0].set_ylabel("Anzahl")
#axs[1].hist(sdnn_afib,300)
#axs[1].set_xlim([0, 300])
#tmp = axs[1].get_ylim()
#axs[1].plot([th_opt,th_opt],[0,10000])
#axs[1].set_ylim(tmp)
#axs[1].set_title("Vorhofflimmern")
#axs[1].set_xlabel("SDNN (ms)")
#axs[1].set_ylabel("Anzahl")
#plt.show()