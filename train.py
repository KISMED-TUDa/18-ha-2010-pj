import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os

fs = 300                                                  # Sampling-Frequenz 300 Hz
detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
sdnn_afib = np.array([])
with open('training/REFERENCE.csv') as csv_file:          # Einlesen der Liste mit Dateinamen und Zuordnung
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      data = sio.loadmat('training/'+row[0]+'.mat')       # Import der EKG-Dateien
      ecg_lead = data['val'][0]
      r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
      sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
      if row[1]=='N':
        sdnn_normal = np.append(sdnn_normal,sdnn)         # Zuordnung zu "Normal"
      if row[1]=='A':
        sdnn_afib = np.append(sdnn_afib,sdnn)             # Zuordnung zu "Vorhofflimmern"
      line_count = line_count + 1
      if (line_count % 100)==0:
        print(str(line_count) + "\t Dateien wurden verarbeitet.")

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