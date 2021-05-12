import os
import copyData
import train_models
import interpret
import sys
import argparse
import evaluate_train

import importlib
importlib.reload(copyData)

def main(args):
    raw_path = "/home/aliarab/scratch/pojects/EEG/raw"
    processed_path = "/home/aliarab/scratch/pojects/EEG/processed"
    batch_size=32
    if args.strip==True:
        processed_path = processed_path + "_stripped"
    if args.aggr==False:
        NumElecs = 60
        model_path = os.path.join(processed_path, "Models/")
        data_path  = os.path.join(processed_path, "h5Files/")
        scores_path = os.path.join(processed_path, "scores/")    
    else:
        NumElecs = 1
        processed_path = processed_path + "_aggr"
        model_path = os.path.join(processed_path, "{}_Models_aggr/".format(args.model))
        data_path  = os.path.join(processed_path, "h5Files_aggr/")
        scores_path = os.path.join(processed_path, "{}_scores_aggr/".format(args.model))
        makeDir(model_path)
        makeDir(data_path)
        makeDir(scores_path)

    #TMS= "LPFC"
    #TEP="LPFC"
    #component= "P180"
    #TMS_TYPE= "ACTIVE"

    TMS = args.TMS
    TEP= args.TEP
    component = args.component
    TMS_TYPE= args.TMS_TYPE
    
    processed_labelPath_parent = os.path.join(processed_path, "TEPS")
    makeDir(processed_labelPath_parent)
    raw_labelPath= os.path.join(raw_path, "TEPS", "SHAM.csv") #this file also includes ACTIVE data
    processed_labelPath = os.path.join(processed_path, "TEPS", "{}_{}_{}.csv".format(TMS_TYPE, TEP, component))
    #copyData.copyLabels(raw_labelPath, processed_labelPath, TEP, component, TMS_TYPE)
    
    raw_features_path_parent = os.path.join(raw_path, "features")
    makeDir(raw_features_path_parent)
    raw_features_path = os.path.join(raw_path, "features", "{}_{}".format(TMS_TYPE, TMS))
    processed_features_path = os.path.join(processed_path , "features", "{}_{}".format(TMS_TYPE, TMS))
    if args.aggr==False:
        print("copying data...")
        #copyData.copyFeatures(raw_features_path, processed_features_path)
    else:
        print("copying data...")
        #copyData.copyFeatures(raw_features_path, processed_features_path, True, True)
    #processed_features_path = "/home/aliarab/scratch/pojects/EEG/processed/features/ACTIVE_LMC_permuted"

    features_repo = processed_features_path
    labels_repo   = processed_labelPath
    data_file = os.path.join(data_path, "norm_{}_{}_{}_{}.h5".format(TMS_TYPE, TEP, component, TMS))
    model_dir = os.path.join(model_path, "norm_{}_{}_{}_{}.pt".format(TMS_TYPE, TEP, component, TMS))
    logPath = os.path.join(model_path, "norm_{}_{}_{}_{}.txt".format(TMS_TYPE, TEP, component, TMS))
    if not os.path.exists(model_dir):
        args = ["--features_repo", features_repo, "--labels_repo", labels_repo,"--data_file", data_file, "--model_dir", model_dir, "--NumElecs", str(NumElecs), "--aggr", "--model", args.model, "--batch_size", str(batch_size)]
        train_models.main_train(args)
    evaluate_train.evaluateModel(data_file, model_dir, batch_size, logPath, NumElecs, 50, 100)

    resultPath = os.path.join(scores_path, "norm_{}_{}_{}_{}_{}.npy".format("FA", TMS_TYPE, TEP, component, TMS))
    interpret.main(model_dir, data_file, resultPath, "IG")


def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--TMS_TYPE', type=str, default='ACTIVE')
    parser.add_argument('--TMS', type=str, default='LPFC')
    parser.add_argument('--component', type=str, default='P30')
    parser.add_argument('--TEP', type=str, default='GLOBAL')
    parser.add_argument("--aggr", action='store_true')
    parser.add_argument("--strip", action='store_true')
    parser.add_argument('--model', type=str, default='TCN')
    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


