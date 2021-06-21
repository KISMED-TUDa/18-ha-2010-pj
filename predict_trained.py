# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Skript testet das neu trainierte Modell


@author: Maurice Rohr
"""

# import socket
# def guard(*args, **kwargs):
#     raise Exception("Internet Access Forbidden")
# socket.socket = guard

from predict import predict_labels
from wettbewerb import load_references, save_predictions
import argparse
import time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')
    args = parser.parse_args()
    
    ecg_leads,ecg_labels,fs,ecg_names = load_references(args.test_dir) # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
    
    start_time = time.time()
    predictions = predict_labels(ecg_leads,fs,ecg_names,use_pretrained=False)
    pred_time = time.time()-start_time
    
    save_predictions(predictions) # speichert Prädiktion in CSV Datei
    print("Runtime",pred_time,"s")