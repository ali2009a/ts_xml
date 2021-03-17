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


def main():
    model_filename ="../Models/m_k_7_model_TCN_NumFeatures_3000.pt"
    pretrained_model = torch.load(open(model_filename, "rb"),map_location=device) 
    pretrained_model.to(device)    
    dataset = data.HDF5Dataset("ts_data.h5")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    arrays = []
    for x_data , target in tqdm(dataloader):
        #data, target = next(iter(dataloader))
        labels =  target.to(device)
        input = x_data.to(device)
        input = Variable(input,  volatile=False, requires_grad=True)
        Data= x_data.reshape(3000, 100).data.cpu().numpy()
        target_=target.data.cpu().numpy()[0]
        Grad = Saliency(pretrained_model)
        attributions = Grad.attribute(input)
        args=""
        saliency_= givenAttGetRescaledSaliency(args,attributions)    
        arrays.append(saliency_)
    np.save("res2.npy", arrays)
    #aggr = np.mean(arrays, axis=0)    
 

def main_plot():
    arrays = np.load("res2.npy")
    aggr = np.mean(arrays, axis=0)
    aggr= aggr.reshape(60,50,100)
    for i in range(6):
        plot(aggr, i*10,i*10+10, "cluster_{}.png".format(i))

def plot(images,s,e, fileName):
    n=e-s
    f, ax = plt.subplots(n,1,sharex=True,figsize=(8,12))
    for i in range(n):
        ax[i].imshow(images[s+i])
    plt.savefig(fileName)
        



