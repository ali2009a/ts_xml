import torch
import torch.optim as optim
import torch.nn.functional as F
import sys, os
sys.path.append("../../Scripts")
from model import Transformer,TCN,LSTMWithInputCellAttention,LSTM
import numpy as np
import argparse
import random
from utils import data_generator
import Helper
from Plotting import plotExampleBox
from torch.autograd import Variable
from  sklearn import preprocessing
import data
import warnings
from  sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

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
    
    model_filename ="../Models/m_k_1_model_TCN_NumFeatures_3000.pt"
    pretrained_model = torch.load(open(model_filename, "rb"),map_location=device) 
    pretrained_model.to(device)    


    dataset = data.HDF5Dataset("ts_data.h5")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    data, target = next(iter(dataloader))
    labels =  target.to(device)
    input = data.to(device)
    input = Variable(input,  volatile=False, requires_grad=True)

    Data=data.reshape(3000, 100).data.cpu().numpy()
    
    target_=target.data.cpu().numpy()[0]
    Grad = Saliency(pretrained_model)
    attributions = Grad.attribute(input)
    args=""
    saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)    
    np.save("res2.npy", saliency_)
 


def plot(images,n, fileName):
    f, ax = plt.subplots(2,1,sharex=True,figsize=(8,12))
    for i in range(n):
        ax[0].imshow(images[i])
    plt.savefig(fileName)
        



