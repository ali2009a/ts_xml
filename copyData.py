import glob
import os
import shutil
import numpy as np
from tqdm import tqdm
import numpy as np
from skimage.measure import block_reduce
import scipy.io
import pandas as pd
from data import pickle_dump
import time

DataPath = "/home/aliarab/scratch/pojects/EEG/raw/"
labelPath  = "/home/aliarab/scratch/pojects/EEG/processed/ALL_TEP_sham.csv"
outPath  = "/home/aliarab/scratch/pojects/EEG/processed/"

electrodes = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']

def main():
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    #copyLabels(True) 
    #copyFeatures()


def copyLabels(labelPath, outPath, TEP, COMPONENT, TMS_TYPE):
    df=pd.read_csv(labelPath)
    df=df[df["TEP"]==TEP]
    df=df[df["COMPONENT"]== COMPONENT]
    df=df[df["TMS_TYPE"]==TMS_TYPE]  #TMS_TYPE could be either SHAM or ACTIVE
    dic={}
    for index, row in df.iterrows():
        dic[(row["ID"], row["TMS"], row["TRIAL"])] = row["AMPLITUDE"]
    pickle_dump(dic, outPath) 
    

def copyFeatures(DataPath, outPath, aggr=False):
    trial_keys=set()
    for subject_folder in glob.glob(os.path.join(DataPath, "*")):
        base_name = os.path.basename(subject_folder)
        file_name = os.path.splitext(base_name)[0]
        tokens = file_name.split("_")
        patient_ID = tokens[0]
        TMS = tokens[1]
        Type=tokens[2]
        electrode = tokens[3]
        trial = int(tokens[4][1:])
        trial_key =(patient_ID, TMS, Type, trial)
        trial_keys.add(trial_key)
    trial_keys = list(trial_keys)
    #for index  in range(0, len(trial_keys)):
    print(trial_keys[:3])
    for index in tqdm(range(0, len(trial_keys))):
        print(index)
        trial_key  = trial_keys[index]
        patient_ID,TMS,Type,trial = trial_key[0],trial_key[1],trial_key[2],trial_key[3]

        isComplete=True
        for elect in electrodes:
            elec_file = os.path.join(DataPath,"{}_{}_{}_{}_T{}.mat".format(patient_ID,TMS,Type, elect, trial))
            if not os.path.exists(elec_file):
                isComplete=False
                print("missing file:")
                print(elec_file)
                continue

        if isComplete==False:
            print("skipped: {}".format(trial_key))
            continue

        folder_name = "{}_{}_{}_T{}".format(patient_ID,TMS,Type, trial)
        folder_path = os.path.join(outPath,folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if aggr==False:
            for elect in electrodes:
                elec_file = os.path.join(DataPath,"{}_{}_{}_{}_T{}.mat".format(patient_ID,TMS,Type, elect, trial))
                elec_data = scipy.io.loadmat(elec_file)
                elec_data = elec_data["HS"]
                elec_data_reduced = block_reduce(elec_data[:,2000:], block_size=(1,10), func=np.mean, cval=np.mean(elec_data[:,2000:]))
                new_subject_dir = os.path.join(folder_path, "{}".format(elect))
                np.save(new_subject_dir, elec_data_reduced, allow_pickle=False)
        else:
            region_electrodes = region2elec(TMS)
            region_elecs = []
            for elect in region_electrodes:
                elec_file = os.path.join(DataPath,"{}_{}_{}_{}_T{}.mat".format(patient_ID,TMS,Type, elect, trial))
                elec_data = scipy.io.loadmat(elec_file)
                elec_data = elec_data["HS"]
                elec_data_reduced = block_reduce(elec_data[:,2000:], block_size=(1,10), func=np.mean, cval=np.mean(elec_data[:,2000:]))
                region_elecs.append(elec_data_reduced)
            region_aggr = np.mean(region_elecs, axis=0)
            new_subject_dir = os.path.join(folder_path, "{}".format("aggr"))
            np.save(new_subject_dir, elec_data_reduced, allow_pickle=False)


def region2elec(region):
    print("region:"+region)
    if region[-1].isnumeric():
        region = region[:-1]
    if region=="LPFC":
        elec = ["F1", "F3", "F5", "F7", "FC1", "FC3", "FC5", "FT7"]
    if region=="RPFC":
        elec = ["F2", "F4", "F6", "F8", "FC2", "FC4", "FC6", "FT8"]
    if region=="LMC":
        elec = ["C1", "C3", "C5", "T7", "CP1", "CP3", "CP5", "TP7"]
    if region=="RMC":
        elec = ["C2", "C4", "C6", "T8", "CP2", "CP4", "CP6", "TP8"]
    if region=="LPC":
        elec = ["P1", "P3", "P5", "P7", "PO3", "PO5", "PO7", "O1"]
    if region=="RMC":
        elec = ["P2", "P4", "P6", "P8", "PO4", "PO6", "PO8", "O2"]
    return elec

def saveFiles(relevant_files, new_subject_dir):        
        arr = np.empty((0,100))
        for fileName in relevant_files:
            trial_data = scipy.io.loadmat(fileName)
            trial_data = trial_data["HS"]
            data_reduced  = arr_reduced = block_reduce(trial_data, block_size=(1,30), func=np.mean, cval=np.mean(trial_data))
            arr=np.concatenate((arr,data_reduced))
        np.save(new_subject_dir, arr)
        #subjectName = os.path.basename(os.path.normpath(subject_folder))
        #if not os.path.exists(new_subject_folder):
        #    os.makedirs(new_subject_folder)


#main()

