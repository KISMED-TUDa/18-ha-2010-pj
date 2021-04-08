import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os

def predictor(model_name):
    with open(model_name, 'rb') as f:
        th_opt = np.load(f)

    if os.path.exists("PREDICTIONS.csv"):
        os.remove("PREDICTIONS.csv")

    fs = 300                                                  # Sampling-Frequenz 300 Hz
    detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

    with open('PREDICTIONS.csv', mode='w', newline='') as predictions_file:
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        with open('test/REFERENCE.csv') as csv_file:          # Einlesen der Liste mit Dateinamen und Zuordnung
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                data = sio.loadmat('test/'+row[0]+'.mat')       # Import der EKG-Dateien
                ecg_lead = data['val'][0]
                r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
                sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden

                if sdnn < th_opt:
                    predictions_writer.writerow([row[0], 'N'])
                else:
                    predictions_writer.writerow([row[0], 'A'])
                line_count = line_count + 1
                if (line_count % 100)==0:
                    print(str(line_count) + "\t Dateien wurden verarbeitet.")