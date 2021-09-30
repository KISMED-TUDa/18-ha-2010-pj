# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 03:07:23 2021

@author: Maurice
"""

from predict import predict_labels
from wettbewerb import load_references, save_predictions
import argparse
import time
from score import score
import datetime
import os, csv

from wki_utilities import Database


'''
INPUT 
Data-Folder (where datasets located)
team_id or team_name
model information like parameters,name, binaryClassifier? (can also be return value of predictions)
which datasets to run (1 and 2, or 3 or 4)

'''

# ### INPUT #TODO delete
# data_folder='./../Datasets/'
# team_id = 1
# datasets_string = '1,2,3,4'
# datasets = [int(i) for i in datasets_string.split(',')]
# model_variants = 'binary' #'multi', 'both'
# model_name = 'ConvNet'
# parameter_dict = {'nr_epochs':20,'alpha':0.1,'n_layers':10} ## Extension
# output_file='error_out'


def wki_evaluate(data_folder,team_id,datasets_string,model_variants,model_name,output_file='error_out',parameter_dict=None):

    datasets = [int(i) for i in datasets_string.split(',')]
    
    db = Database()
    nr_runs = db.get_nr_runs(team_id)
    if nr_runs==None:
        new_nr_runs=1
    else:
        new_nr_runs = nr_runs+1
    run_times = dict()
    run_successfull = dict()
    
    # TODO reconfigure output to file
    
    if model_variants == 'binary':
        variant_bools=[True]
    if model_variants == 'multi':
        variant_bools=[False]
    else:
        variant_bools = [False,True]
        
    for is_binary_classifier in variant_bools:    
        
        model_id = db.put_model(team_id,model_name,is_binary_classifier,parameter_dict=None)
        
        for dataset_id in datasets:
            dataset_folder =  db.get_dataset_folder(dataset_id)  
            
            ### make predictions & measure time
            
            ecg_leads,ecg_labels,fs,ecg_names = load_references(os.path.join(data_folder, dataset_folder)) # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
            start_time = time.time()
            try:
                predictions = predict_labels(ecg_leads,fs,ecg_names,model_name=model_name,is_binary_classifier=is_binary_classifier)
                #TODO check predictions for plausibility
                run_successfull[dataset_id]=True
                print('Prediction SUCCESSFULL')
                print(f"(for model {model_name} and team_id {team_id})")
            except Exception as e:
                #print something
                print(e)
                run_successfull[dataset_id]=False
            finally:
                run_times[dataset_id]=int(time.time()-start_time)
                print("Runtime",run_times[dataset_id],"s")
            
            if run_successfull[dataset_id]:
                save_predictions(predictions) # speichert Prädiktion in CSV Datei
                ### compute scores and save to database
                F1,F1_mult,Conf_Matrix = score(os.path.join(data_folder, dataset_folder))
        
                
                db.put_scored_entry(dataset_id,team_id,new_nr_runs,F1,F1_mult,model_id,run_times[dataset_id],Conf_Matrix)
            else:
                db.put_unscored_entry(dataset_id,team_id,model_id,run_times[dataset_id],output_file)
                
                
                

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Score based on Predictions and put in database')
    parser.add_argument('data_folder', action='store',type=str)
    parser.add_argument('team_id', action='store',type=int)
    parser.add_argument('--datasets', action='store',type=str,default='1,2')
    parser.add_argument('--model_variants', action='store',type=str,default='multi')
    parser.add_argument('--model_name', action='store',type=str,default='dummy')        

    args = parser.parse_args()
    
    wki_evaluate(args.data_folder,args.team_id,args.datasets,args.model_variants,args.model_name,output_file='error_out',parameter_dict=None)               
