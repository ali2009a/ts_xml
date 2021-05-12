import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys, os
from model import TCN, LSTM, Transformer, LSTMWithInputCellAttention
import numpy as np
import argparse
from tqdm import tqdm
import data
from torchsummary import summary
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")

import importlib
importlib.reload(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models=["TCN"]


def createDataFile(args):
    print("creating file...")
    if not os.path.exists(args.data_file) or True:
        print("Writing the images to h5 file...")
        training_files, subject_ids = data.fetch_training_data_files(args.features_repo, aggr=args.aggr)
        truthData = data.getTruthData(subject_ids, args.labels_repo)
        data.write_data_to_file(training_files, args.data_file, image_shape=[args.NumFeatures, args.NumTimeSteps], subject_ids=subject_ids, truthData=truthData)

def trainFold(args, k, train_loader, val_loader, test_loader):
    print("k:{}".format(k))
    m = "TCN"
    channel_sizes = [args.nhid] * args.levels
    model = TCN(args.NumElecs*args.NumFeatures, args.n_classes, channel_sizes, kernel_size=args.ksize, dropout=args.dropout)
    summary(model, (args.NumElecs*args.NumFeatures, args.NumTimeSteps))
    model.to(device)
    model_name = "k_{}_model_{}_NumFeatures_{}".format(k, m, args.NumFeatures*args.NumElecs)
    model_filename = args.model_dir + 'm_' + model_name + '.pt'
    lr=args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    best_test_loss=100
    for epoch in range(1, args.epochs+1):
        print("epoch: {}".format(epoch))
        model,optimizer = train(args,epoch,model,train_loader,optimizer)
        print("validation set:::")
        test_loss,test_acc = test(args,model,val_loader)
        print("test set::::")
        #test(args,model,test_loader)
        if(test_loss<best_test_loss):
            best_test_loss = test_loss
            save(model, model_filename)
        if(test_acc>=0.9):
            break
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def main_k(args):
    #createDataFile(args)
    k=10
    for i in range(0,10):
        train_loader, val_loader, test_loader = data.generator(args.data_file, batch_size=args.batch_size, validation_split=0.1, kFold=k, fold=i, allowShare=True, shuffle=True)
        trainFold(args, i, train_loader, val_loader, test_loader)


def main(args):
    torch.manual_seed(args.seed)
    if not os.path.exists(args.data_file):
        print("Writing the images to h5 file...")
        training_files, subject_ids = data.fetch_training_data_files(args.features_repo, args.aggr)
        truthData = data.getTruthData(subject_ids, args.labels_repo)
        data.write_data_to_file(training_files, args.data_file, image_shape=[args.NumFeatures, args.NumTimeSteps], subject_ids=subject_ids, truthData=truthData, n_channels = args.NumElecs)
        data.normalizeDataset(args.data_file, args.NumFeatures, args.NumElecs, args.NumTimeSteps)
    else:
        print("data file exists. Loading existing one...")
    
    train_loader, val_loader, test_loader = data.generator(args.data_file, batch_size=args.batch_size, validation_split=0.1, kFold=10, fold=0, allowShare=True, shuffle=True)
    m = "TCN"
    channel_sizes = [args.nhid] * args.levels
    if args.model == "TCN":
        print("using TCN model")
        model = TCN(args.NumElecs*args.NumFeatures, args.n_classes, channel_sizes, kernel_size=args.ksize, dropout=args.dropout)
    elif args.model == "LSTM":
        print("using LSTM model")
        model = LSTM(args.NumElecs*args.NumFeatures,  args.nhid, args.n_classes,args.dropout)
    else:
        print("invalid model!")
    #summary(model, (args.NumElecs*args.NumFeatures, args.NumTimeSteps))
    model.to(device)
    #model_name = "model_{}_NumFeatures_{}".format(m,args.NumFeatures*60)
    #model_filename = args.model_dir + 'm_' + model_name + '.pt'
    model_filename = args.model_dir
    lr=args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    best_test_loss=100
    #best_test_acc = -1
    test_loss,test_acc = test(args,model,val_loader)
    print("random stats: loss:{}, ev: {}".format(test_loss, test_acc))
    for epoch in range(1, args.epochs+1):
        print("epoch: {}".format(epoch))
        model,optimizer = train(args,epoch,model,train_loader,optimizer)
        test_loss,test_acc = test(args,model,val_loader)
        if(test_loss<best_test_loss):
        #if(test_acc> best_test_acc):
            best_test_loss = test_loss
            #best_test_acc = test_acc
            save(model, model_filename)
        if(test_acc>=0.9):
            break
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def save(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)


def train(args,ep,model,train_loader,optimizer):
    train_loss = 0
    model.train()
    print("Number of batches: {}".format(len(train_loader)))
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, args.NumElecs*args.NumFeatures, args.NumTimeSteps)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.smooth_l1_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            message = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval))
            print(message)
            train_loss = 0
    return model, optimizer


def test(args,model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    Acc=0
    total_targets= np.array([])
    total_predictions = np.array([])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = data.view(-1, args.NumElecs*args.NumFeatures, args.NumTimeSteps)
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
        message = ('\nValidation Set: Average loss: {:.10f}, ev: {})\n'.format(
            test_loss, Acc))
        print(message)
        return test_loss,Acc

def parse_arguments(raw_args):
    parser = argparse.ArgumentParser()


    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout applied to layers (default: 0.05)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit (default: 20)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')


    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='report interval (default: 1')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate (default: 2e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--nhid', type=int, default=25,
                        help='number of hidden units per layer (default: 25)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')

    parser.add_argument('--n_classes', type=int, default=1)


    parser.add_argument('--data_dir', type=str, default="../Data/")
    parser.add_argument('--model_dir', type=str, default="/home/aliarab/scratch/pojects/EEG/wd/Models/")


    parser.add_argument('--NumTimeSteps',type=int,default=100)
    parser.add_argument('--NumFeatures',type=int,default=50)
    parser.add_argument('--NumElecs',type=int,default=60)

    parser.add_argument('--n_layers', type=int, default=60)
    parser.add_argument('--heads', type=int, default=4)

    parser.add_argument('--attention_hops', type=int, default=28)
    parser.add_argument('--d_a', type=int, default=30)

    parser.add_argument('--data_file', type=str, default="ts_data.h5")
    parser.add_argument('--features_repo', type=str, default="/home/aliarab/scratch/pojects/EEG/processed/LPFC")
    parser.add_argument('--labels_repo', type=str, default="data/original/labels.pkl")

    parser.add_argument("--writeOnly", action='store_true')
    parser.add_argument("--sham", action='store_true')
    parser.add_argument("--aggr", action='store_true')
    parser.add_argument('--model', type=str, default="TCN")
    return  parser.parse_args(raw_args)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main_k(args)


def main_train(raw_args):
    args = parse_arguments(raw_args)
    print("args.aggr:")
    print(args.aggr)
    main(args) 

    
