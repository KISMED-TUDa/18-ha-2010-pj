# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:00:13 2021

@author: Maurice
"""

from score import score
import argparse
import datetime
import os, csv


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Score based on Predictions')
    parser.add_argument('output_dir', action='store',type=str)
    parser.add_argument('data_set', action='store',type=str)
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')

    args = parser.parse_args()
    
    current_time = datetime.datetime.now()
    F1,F1_mult,Conf_Matrix = score(args.test_dir)
    with open('teamname.txt') as team_file:
        teamname = team_file.readline()[:-1]
        
    filename =   args.output_dir + teamname + '.csv'  
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as scores_file:
            scores_writer = csv.writer(scores_file, delimiter=';')
            scores_writer.writerow(['team_name','date_time','dataset','F1_score','multilabel_score'])
    
    with open(filename, mode='+a', newline='') as scores_file:
        scores_writer = csv.writer(scores_file, delimiter=';')
        
        scores_writer.writerow([teamname,str(current_time), args.data_set, F1, F1_mult] )
    with open(args.output_dir+teamname+str(current_time.date())+'_'+ str(current_time.hour)+'_'+str(current_time.minute) + '.csv',mode='w',newline='') as matrix_file:
        matrix_writer = csv.writer(matrix_file, delimiter=';')
        true_names = list(Conf_Matrix.keys())
        pred_names = list(Conf_Matrix[true_names[0]].keys())
        matrix_writer.writerow([''] + pred_names)
        for tn in true_names:
            matrix_writer.writerow([tn] + list(Conf_Matrix[tn].values()))

        