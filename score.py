import os
import sys
import pandas as pd

if not os.path.exists("PREDICTIONS.csv"):
    sys.exit("Es gibt keine Predictions")  

if  not os.path.exists("test/REFERENCE.csv"):
    sys.exit("Es gibt keine Ground Truth")  

df_pred = pd.read_csv("PREDICTIONS.csv", header=None)
df_gt = pd.read_csv("test/REFERENCE.csv", header=None)

N_files = df_gt.shape[0]

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(N_files):
    gt_name = df_gt[0][i]
    gt_class = df_gt[1][i]

    pred_indx = df_pred[df_pred[0]==gt_name].index.values

    if not pred_indx.size:
        print("Prediktion für " + gt_name + " fehlt, nehme \"normal\" an.")
        pred_class = "N"
    else:
        pred_indx = pred_indx[0]
        pred_class = df_pred[1][pred_indx]

    if gt_class == "A" and pred_class == "A":
        TP = TP + 1
    if gt_class == "N" and pred_class == "N":
        TN = TN + 1
    if gt_class == "N" and pred_class == "A":
        FP = FP + 1
    if gt_class == "A" and pred_class == "N":
        FN = FN + 1


F1 = TP / (TP + 1/2*(FP+FN))
print(F1)