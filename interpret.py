import torch
import torch.optim as optim
import torch.nn.functional as F
import sys, os
import numpy as np
import argparse
import random
from torch.autograd import Variable
from  sklearn import preprocessing
import data
import warnings
from  sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
warnings.filterwarnings("ignore")

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation,
    Occlusion

)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def givenAttGetRescaledSaliency(args,attributions,isTensor=True):
    if(isTensor):
        saliency = np.absolute(attributions.data.cpu().numpy())
    else:
        saliency = np.absolute(attributions)
    #saliency=saliency.reshape(-1,args.NumTimeSteps*args.NumFeatures)
    saliency=saliency.reshape(-1, 3000*100)
    rescaledSaliency=minmax_scale(saliency,axis=1)
    rescaledSaliency=rescaledSaliency.reshape(attributions.shape)
    return rescaledSaliency


def main(model_filename, datafile, resultPath, method="FA"):
    #model_filename ="../Models/m_k_0_model_TCN_NumFeatures_3000.pt"
    pretrained_model = torch.load(open(model_filename, "rb"),map_location=device)
    pretrained_model.to(device)    
    dataset = data.HDF5Dataset(datafile)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    arrays = []
    FA = FeatureAblation(pretrained_model)
    Grad = Saliency(pretrained_model)
    Grad_ = Saliency(pretrained_model)
    SG = NoiseTunnel(Grad_)
    IG = IntegratedGradients(pretrained_model)

    for index, (x_data , target) in tqdm(enumerate(dataloader)):
        print(index)
        saliency_ = processImage(x_data, target, FA, method)
        arrays.append(saliency_)
        np.save(resultPath, arrays)
 
def processImage(x_data, target, FA, method):
    labels =  target.to(device)
    input = x_data.to(device)
    input = Variable(input,  volatile=False, requires_grad=True)
    Data= x_data.reshape(3000, 100).data.cpu().numpy()
    target_=target.data.cpu().numpy()[0]
    args=""
    baseline_single=torch.Tensor(np.random.random(input.shape)).to(device)
    
    #attributions = Grad.attribute(input)
    #saliency_= givenAttGetRescaledSaliency(args,attributions)    

    if method=="FA":
        attributions = FA.attribute(input)
        saliency_= givenAttGetRescaledSaliency(args,attributions)

    #attributions = SG.attribute(input)        
    #saliency_= givenAttGetRescaledSaliency(args,attributions)

    if method=="IG":
        #attributions = IG.attribute(input, baselines=baseline_single)
        #saliency_= givenAttGetRescaledSaliency(args,attributions)
        TSR_attributions =getTwoStepRescaling(IG, input, 100, 3000, labels,hasBaseline=baseline_single)
        saliency_= givenAttGetRescaledSaliency(args, TSR_attributions, isTensor=False)
    return saliency_


def getTwoStepRescaling(Grad, input, sequence_length,input_size, TestingLabel,hasBaseline=None,hasFeatureMask=None,hasSliding_window_shapes=None):
    assignment=input[0,0,0]
    timeGrad=np.zeros((1,sequence_length))
    inputGrad=np.zeros((input_size,1))
    newGrad=np.zeros((input_size, sequence_length))
    if(hasBaseline==None):  
        ActualGrad = Grad.attribute(input,target=TestingLabel).data.cpu().numpy()
    else:
        if(hasFeatureMask!=None):
            ActualGrad = Grad.attribute(input,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
        elif(hasSliding_window_shapes!=None):
            ActualGrad = Grad.attribute(input,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(input,baselines=hasBaseline).data.cpu().numpy()
    for t in range(sequence_length):
        timeGrad[:,t] = np.mean(np.absolute(ActualGrad[0,:,t]))
    timeContibution=preprocessing.minmax_scale(timeGrad, axis=1)
    meanTime = np.quantile(timeContibution, .55)        
    for t in range(sequence_length):
        print(t)
        if(timeContibution[0,t]>meanTime):
            for c in range(input_size):
                newInput = input.clone()
                newInput[:,c,t]=assignment
                if(hasBaseline==None):  
                    inputGrad_perInput = Grad.attribute(newInput).data.cpu().numpy()
                else:
                    if(hasFeatureMask!=None):
                        inputGrad_perInput = Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
                    elif(hasSliding_window_shapes!=None):
                        inputGrad_perInput = Grad.attribute(newInput,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
                    else:
                        inputGrad_perInput = Grad.attribute(newInput,baselines=hasBaseline).data.cpu().numpy()
                inputGrad_perInput=np.absolute(ActualGrad - inputGrad_perInput)
                inputGrad[c,:] = np.sum(inputGrad_perInput)
            featureContibution= preprocessing.minmax_scale(inputGrad, axis=0)
        else:
            featureContibution=np.ones((input_size,1))*0.1
        for c in range(input_size):
           newGrad [c,t]= timeContibution[0,t]*featureContibution[c,0]
    return newGrad




def main_plot(input_file, output_path, prefix):
    arrays = np.load(input_file)
    aggr = np.mean(arrays, axis=0)
    aggr= aggr.reshape(60,50,100)
    for i in range(6):
        plot(aggr, i*10,i*10+10, "{}/{}_cluster_{}.png".format(output_path, prefix, i))


def main_plot_merged(input_file, output_path, array_path):
    arrays = np.load(input_file)
    aggr = np.mean(arrays, axis=0)
    aggr= aggr.reshape(60,50,100)
    aggr= np.mean(aggr, axis=0)
    plt.imshow(aggr, origin='lower')
    #np.save(array_path, aggr)
    np.savetxt(array_path, aggr, delimiter=",")
    #fileName="{}_merged.png".format(prefix)
    plt.savefig(output_path)


def visualizeScores(scores_path, outputPath):
    for subject_dir in glob.glob(os.path.join(scores_path, "*")):
        if not os.path.isfile(subject_dir):
            continue
        print(subject_dir)
        parent= os.path.dirname(subject_dir)
        fileName=os.path.basename(subject_dir)
        base, extension = os.path.splitext(fileName)
        main_plot_merged(subject_dir, os.path.join(outputPath,"merged_{}.png".format(base)), os.path.join(outputPath,"merged_{}.csv".format(base)))

def visualizeScores2(scores_path, outputPath):
    for subject_dir in glob.glob(os.path.join(scores_path, "*")):
        if not os.path.isfile(subject_dir):
            continue
        print(subject_dir)
        parent= os.path.dirname(subject_dir)
        fileName=os.path.basename(subject_dir)
        base, extension = os.path.splitext(fileName)
        main_plot(subject_dir, outputPath, base)



def plot(images,s,e, fileName):
    n=e-s
    f, ax = plt.subplots(n,1,sharex=True,figsize=(8,12))
    for i in range(n):
        ax[i].imshow(images[s+i], origin='lower')
    plt.savefig(fileName)
    


def plotSamples(fileName, output_file):
    dataset = data.HDF5Dataset("ts_data.h5")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    indices = np.random.choice(list(range(2000)), 10)
    f, ax = plt.subplots(10,1,sharex=True, figsize=(10,12))
    i=0
    for index, (x_data , target) in tqdm(enumerate(dataloader)):
        if index in indices:
            Data= x_data.reshape(3000, 100).data.cpu().numpy()
            Data= Data.reshape(60,50,100)
            ax[i].imshow(Data[0,:,:], origin='lower') 
            i=i+1
    plt.savefig(output_file)    

if __name__ == '__main__':
    main()       



