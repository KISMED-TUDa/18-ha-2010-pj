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
from wettbewerb import load_references

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)


detectors = Detectors(fs)                                     # Initialisierung des QRS-Detektors

sdnn_normal_ham = np.array([])                                # Hamilton normal Array
sdnn_afib_ham = np.array([])                                  # Hamilton afib Array

sdnn_normal_chr = np.array([])                                # Christov normal Array
sdnn_afib_chr = np.array([])                                  # Christov afib Array

sdnn_normal_eng = np.array([])                                # Eng normal Array
sdnn_afib_eng = np.array([])                                  # Eng afib Array

sdnn_normal_swt = np.array([])                           
sdnn_afib_swt = np.array([])                                 

sdnn_normal_pt = np.array([])                               
sdnn_afib_pt = np.array([])                                  

sdnn_normal_ta = np.array([])                             
sdnn_afib_ta = np.array([])                                

for idx, ecg_lead in enumerate(ecg_leads):
    r_peaks_ham = detectors.hamilton_detector(ecg_lead)         # Detektion der QRS-Komplexe mit Hamilton Detector
    r_peaks_chr = detectors.christov_detector(ecg_lead)         # Detektion der QRS-Komplexe mit Christov Detector
    #r_peaks_eng = detectors.engzee_detector(ecg_lead)           # fehlerhaft?
    r_peaks_swt = detectors.swt_detector(ecg_lead)              # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector
    r_peaks_pt = detectors.pan_tompkins_detector(ecg_lead)      # Detektion der QRS-Komplexe mit Pan Tompkins Detector
    r_peaks_ta = detectors.two_average_detector(ecg_lead)       # Detektion der QRS-Komplexe mit Two Average Detector
    
    sdnn_ham = np.std(np.diff(r_peaks_ham)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Hamilton
    sdnn_chr = np.std(np.diff(r_peaks_chr)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Christov
    #sdnn_eng = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Engzee
    sdnn_swt = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Stationary Wavelet Transform
    sdnn_pt = np.std(np.diff(r_peaks_pt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Pan Tompkins
    sdnn_ta = np.std(np.diff(r_peaks_ta)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Two Average


    if ecg_labels[idx] == 'N':
        sdnn_normal_ham = np.append(sdnn_normal_ham, sdnn_ham)         # Zuordnung zu "Normal"
        sdnn_normal_chr = np.append(sdnn_normal_chr, sdnn_chr)
        #sdnn_normal_eng = np.append(sdnn_normal_eng, sdnn_eng)
        sdnn_normal_swt = np.append(sdnn_normal_swt, sdnn_swt)
        sdnn_normal_pt = np.append(sdnn_normal_pt, sdnn_pt)
        sdnn_normal_ta = np.append(sdnn_normal_ta, sdnn_ta)
    if ecg_labels[idx] == 'A':
        sdnn_afib_ham = np.append(sdnn_afib_ham, sdnn_ham)             # Zuordnung zu "Vorhofflimmern"
        sdnn_afib_chr = np.append(sdnn_afib_chr, sdnn_chr) 
        #sdnn_afib_eng = np.append(sdnn_afib_eng, sdnn_eng)
        sdnn_afib_swt = np.append(sdnn_afib_swt, sdnn_swt)
        sdnn_afib_pt = np.append(sdnn_afib_pt, sdnn_pt)
        sdnn_afib_ta = np.append(sdnn_afib_ta, sdnn_ta)

    if (idx % 100) == 0:
        print(str(idx) + "\t EKG Signale wurden verarbeitet.")
        
################################################### PLOT

fig, axs = plt.subplots(2, 6, constrained_layout=True, figsize=(15, 15))
                                                                # ham
axs[0][0].hist(sdnn_normal_ham,2000)
axs[0][0].set_xlim([0, 300])
axs[0][0].set_title("Normal - Hamilton")
axs[0][0].set_xlabel("SDNN (ms)")
axs[0][0].set_ylabel("Anzahl")
axs[1][0].hist(sdnn_afib_ham,300)
axs[1][0].set_xlim([0, 300])
axs[1][0].set_title("Vorhofflimmern - Hamilton")
axs[1][0].set_xlabel("SDNN (ms)")
axs[1][0].set_ylabel("Anzahl")
                                                                # chr
axs[0][1].hist(sdnn_normal_chr,2000)
axs[0][1].set_xlim([0, 300])
axs[0][1].set_title("Normal - Christov")
axs[0][1].set_xlabel("SDNN (ms)")
axs[0][1].set_ylabel("Anzahl")
axs[1][1].hist(sdnn_afib_chr,300)
axs[1][1].set_xlim([0, 300])
axs[1][1].set_title("Vorhofflimmern - Christov")
axs[1][1].set_xlabel("SDNN (ms)")
axs[1][1].set_ylabel("Anzahl")
                                                                # eng
axs[0][2].hist(sdnn_normal_eng,2000)
axs[0][2].set_xlim([0, 300])
axs[0][2].set_title("Normal - eng")
axs[0][2].set_xlabel("SDNN (ms)")
axs[0][2].set_ylabel("Anzahl")
axs[1][2].hist(sdnn_afib_eng,300)
axs[1][2].set_xlim([0, 300])
axs[1][2].set_title("Vorhofflimmern - eng")
axs[1][2].set_xlabel("SDNN (ms)")
axs[1][2].set_ylabel("Anzahl")
                                                                # swt
axs[0][3].hist(sdnn_normal_swt,2000)
axs[0][3].set_xlim([0, 300])
axs[0][3].set_title("Normal - swt")
axs[0][3].set_xlabel("SDNN (ms)")
axs[0][3].set_ylabel("Anzahl")
axs[1][3].hist(sdnn_afib_swt,300)
axs[1][3].set_xlim([0, 300])
axs[1][3].set_title("Vorhofflimmern - swt")
axs[1][3].set_xlabel("SDNN (ms)")
axs[1][3].set_ylabel("Anzahl")
                                                                # pt
axs[0][4].hist(sdnn_normal_pt,2000)
axs[0][4].set_xlim([0, 300])
axs[0][4].set_title("Normal - pt")
axs[0][4].set_xlabel("SDNN (ms)")
axs[0][4].set_ylabel("Anzahl")
axs[1][4].hist(sdnn_afib_pt,300)
axs[1][4].set_xlim([0, 300])
axs[1][4].set_title("Vorhofflimmern - pt")
axs[1][4].set_xlabel("SDNN (ms)")
axs[1][4].set_ylabel("Anzahl")
                                                                # ta
axs[0][5].hist(sdnn_normal_ta,2000)
axs[0][5].set_xlim([0, 300])
axs[0][5].set_title("Normal - ta")
axs[0][5].set_xlabel("SDNN (ms)")
axs[0][5].set_ylabel("Anzahl")
axs[1][5].hist(sdnn_afib_ta,300)
axs[1][5].set_xlim([0, 300])
axs[1][5].set_title("Vorhofflimmern - ta")
axs[1][5].set_xlabel("SDNN (ms)")
axs[1][5].set_ylabel("Anzahl")

plt.show()