import torch
import torch.optim as optim
import torch.nn.functional as F
import sys, os
sys.path.append("../../Scripts")
import numpy as np
import argparse
import random
from torch.autograd import Variable
from  sklearn import preprocessing
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
import data
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

models=  ["TCN"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def drawPlot(x,y, title, xLabel, yLabel, fileName):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x, y)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    plt.savefig(fileName)

def main_k(args):
    k=10
    val_file = open("val_stat.csv", "w")
    test_file = open("test_stat.csv", "w")
    train_file = open("train_stat.csv", "w")   
    val_file.write("MSE, r2, ev\n")
    train_file.write("MSE, r2, ev\n")
    test_file.write("MSE, r2, ev\n")
    for i in range(0,k):
        train_loader, val_loader, test_loader = data.generator(args.data_dir, batch_size=1, validation_split=0.1, kFold=k, fold=i, allowShare=True, shuffle=True)
        for m in range(len(models)):
            print("k:{}".format(i))
            model_name = "k_{}_model_{}_NumFeatures_{}".format(i, models[m],args.NumFeatures)
            model_filename = args.model_dir + 'm_' + model_name + '.pt'
            pretrained_model = torch.load(open(model_filename, "rb"),map_location=device)
            pretrained_model.to(device)

            MSE, r2, ev = computeMeasures(val_loader, pretrained_model, args, "val.png")
            val_file.write("{}, {}, {}\n".format(MSE, r2, ev))

            MSE, r2, ev = computeMeasures(test_loader, pretrained_model, args, "test.png")
            test_file.write("{}, {}, {}\n".format(MSE, r2, ev))

            MSE, r2, ev = computeMeasures(train_loader, pretrained_model, args, "train.png")
            train_file.write("{}, {}, {}\n".format(MSE, r2, ev))
    val_file.close()
    test_file.close()
    train_file.close()


def main_sham(args):
    sham_file = open("sham_stat.csv", "w")
    dataset = data.HDF5Dataset(args.data_dir)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    k=10
    for i in range(k):
        model_name = "k_{}_model_{}_NumFeatures_{}".format(i, models[0], args.NumFeatures)
        model_filename = args.model_dir + 'm_' + model_name + '.pt'
        pretrained_model = torch.load(open(model_filename, "rb"),map_location=device)
        pretrained_model.to(device)
        MSE, r2, ev = computeMeasures(dataloader, pretrained_model, args, "sham.png")
        sham_file.write("{}, {}, {}\n".format(MSE, r2, ev))
    sham_file.close()

def main_stat_test(args):
    stat_file = open("stat_test.csv", "w")
    dataset_sham = data.HDF5Dataset("ts_data_sham.h5")
    dataset_active = data.HDF5Dataset("ts_data.h5")
    dataloader_sham = data.DataLoader(dataset_sham, batch_size=1, shuffle=False, num_workers=0)
    dataloader_active = data.DataLoader(dataset_active, batch_size=1, shuffle=False, num_workers=0)
    k=10
    for i in range(k):
        print(i)
        model_name = "k_{}_model_{}_NumFeatures_{}".format(i, models[0], args.NumFeatures)
        model_filename = args.model_dir + 'm_' + model_name + '.pt'
        pretrained_model = torch.load(open(model_filename, "rb"),map_location=device)
        pretrained_model.to(device)
        print("sham...")
        MSE_sham, r2_sham, ev_sham, sham_target, sham_prediction = computeMeasures(dataloader_sham, pretrained_model, args, "sham.png", return_residuals=True)
        sham_res = sham_target- sham_prediction
        sham_array = np.array([sham_target, sham_prediction])
        df_sham = pd.DataFrame(sham_array.T, columns= ["target","prediction"])
        df_sham.to_csv("sham_df_{}.csv".format(i))
        print("active...")
        MSE_active, r2_active, ev_active, active_target, active_prediction = computeMeasures(dataloader_active, pretrained_model, args, "active.png", return_residuals=True)
        active_res = active_target - active_prediction
        active_array = np.array([active_target, active_prediction])
        df_active = pd.DataFrame(active_array.T, columns= ["target","prediction"])
        df_active.to_csv("active_df_{}.csv".format(i))
        t,p =stats.ttest_ind(sham_res, active_res)
        stat_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(MSE_active, MSE_sham, r2_active, r2_sham, ev_active, ev_sham, t, p, np.mean(np.abs(active_res)), np.mean(np.abs(sham_res))))
    stat_file.close()

def main(args):
    train_loader, val_loader, test_loader = data.generator(args.data_dir, batch_size=1, train_split=0.9, validation_split=0.05)
    for m in range(len(models)):
        model_name = "model_{}_NumFeatures_{}".format(models[m],args.NumFeatures)
        model_filename = args.model_dir + 'm_' + model_name + '.pt'
        pretrained_model = torch.load(open(model_filename, "rb"),map_location=device) 
        pretrained_model.to(device)


        MSE, r2, ev = computeMeasures(val_loader, pretrained_model, args, "val.png")
        print("test: MSE:{}, r2:{}, ev:{}".format(MSE, r2, ev))

        MSE, r2, ev = computeMeasures(test_loader, pretrained_model, args, "test.png")
        print("test: MSE:{}, r2:{}, ev:{}".format(MSE, r2, ev))

        MSE, r2, ev = computeMeasures(train_loader, pretrained_model, args, "train.png")
        print("train: MSE:{}, r2:{}, ev:{}".format(MSE, r2, ev))


def evaluateModel(data_file, model_filename, batch_size, logPath, NumElecs, NumFeatures, NumTimeSteps):
    train_loader, val_loader, test_loader = data.generator(data_file, batch_size=batch_size, validation_split=0.1, kFold=10, fold=0, allowShare=True, shuffle=True)
    pretrained_model = torch.load(open(model_filename, "rb"),map_location=device)
    pretrained_model.to(device)

    with open(logPath, "w") as f:
        MSE, r2, ev = computeMeasures(val_loader, pretrained_model, "val.png", NumElecs, NumFeatures, NumTimeSteps)
        f.write("validation set -   MSE:{}, r2:{}, ev:{}\n".format(MSE, r2, ev))

        MSE, r2, ev = computeMeasures(test_loader, pretrained_model, "test.png", NumElecs, NumFeatures, NumTimeSteps)
        f.write("test set       -   MSE:{}, r2:{}, ev:{}\n".format(MSE, r2, ev))

        MSE, r2, ev = computeMeasures(train_loader, pretrained_model, "train.png", NumElecs, NumFeatures, NumTimeSteps)
        f.write("train set      -   MSE:{}, r2:{}, ev:{}\n".format(MSE, r2, ev))
     

 
def computeMeasures(data_loader, model, fileName, NumElecs, NumFeatures, NumTimeSteps,  return_residuals=False):
    loss, acc, target, predictions = test( model, data_loader, NumElecs, NumFeatures, NumTimeSteps)
    MSE, r2, ev = computeMetrics(target, predictions)      
    #drawPlot(target-predictions, predictions, "Residual Plot", "TEP Predictions (Microvolt)", "Residual", fileName )
    if return_residuals:
        return [MSE, r2, ev, target, predictions]
    else:
        return [MSE, r2, ev]

def computeMetrics(y,t):
    MSE = metrics.mean_squared_error(t,y)
    r2 = metrics.r2_score(y,t)
    ev = metrics.explained_variance_score(y,t)
    return MSE, r2, ev


#def test(args,model,test_loader):
#    model.eval()
#    test_loss = 0
#    correct = 0
#    Acc=0
#    total_targets= np.array([])
#    total_predictions = np.array([])
#    with torch.no_grad():
#        for data, target in test_loader:
#            data, target = data.to(device), target.to(device)
#
#            data = data.view(-1, 3000, args.NumTimeSteps)
#            data, target = Variable(data, volatile=True), Variable(target)
#            output = model(data)
#            test_loss += F.smooth_l1_loss(output, target, size_average=False).item()
#            total_targets = np.concatenate((total_targets,target.detach().cpu().numpy().reshape((-1))))
#            total_predictions = np.concatenate((total_predictions, output.detach().cpu().numpy().reshape((-1))))
#
#        test_loss /= len(test_loader.dataset)
#        #Acc = 100. * correct / len(test_loader.dataset)
#        Acc = metrics.explained_variance_score(total_targets.reshape((-1)), total_predictions.reshape((-1)))
#        message = ('\nTest set: Average loss: {:.10f}, accuracy: {})\n'.format(
#            test_loss, Acc))
#        print(message)
#        return test_loss, Acc, total_targets, total_predictions
#

def test(model,test_loader, NumElecs, NumFeatures, NumTimeSteps):
    model.eval()
    test_loss = 0
    correct = 0
    Acc=0
    total_targets= np.array([])
    total_predictions = np.array([])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = data.view(-1, NumElecs*NumFeatures, NumTimeSteps)
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.smooth_l1_loss(output, target, size_average=False).item()
            total_targets = np.concatenate((total_targets,target.detach().cpu().numpy().reshape((-1))))
            total_predictions = np.concatenate((total_predictions, output.detach().cpu().numpy().reshape((-1))))

        test_loss /= len(test_loader.dataset)
        #Acc = 100. * correct / len(test_loader.dataset)
        total_targets = total_targets.reshape((-1))
        total_predictions = total_predictions.reshape((-1))
        Acc = metrics.explained_variance_score(total_targets, total_predictions)
        message = ('\nAverage loss: {:.10f}, ev: {})\n'.format(
            test_loss, Acc))
        print(message)
        return test_loss,Acc, total_targets, total_predictions



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/home/aliarab/scratch/pojects/EEG/wd/evaluation_sham/')


    parser.add_argument('--data_dir', type=str, default="ts_data_sham.h5")
    parser.add_argument('--model_dir', type=str, default="../Models/")


    parser.add_argument('--NumTimeSteps',type=int,default=100)
    parser.add_argument('--NumFeatures',type=int,default=3000)

    return  parser.parse_args()

if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    main_k(parse_arguments(sys.argv[1:]))
    #main_stat_test(parse_arguments(sys.argv[1:]))
