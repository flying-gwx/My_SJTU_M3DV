#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import pandas as pd
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
#import ipdb




#import shutil

#import setproctitle
import befortraining as be
import densenet
#import make_graph

def train(cuda, epoch, net, trainLoader, optimizer,out):
    net.train()
    
    
    nTrain = len(trainLoader.dataset)
    incorrect=0
    ave_loss=0
    
    for batch_idx, value in enumerate(trainLoader):
        data,target = value['input'].unsqueeze(1), value['label'].squeeze(1)
        
        if cuda:
            data, target = data.cuda(), target.cuda()
#        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        
        output = net(data)
        
        
        loss = F.cross_entropy(output,target)
        ave_loss+=loss.item()
       
        
        loss.backward()
        
        optimizer.step()
        
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.view_as(pred)).cpu().sum().item()

        
    err=100.*incorrect/nTrain
    ave_loss/=nTrain
    out.write('{:.5f}\n'.format(ave_loss))
    
        
    print('Train Epoch: {:.2f} \tLoss: {:.6f}\tError: {:.6f}'.format(
            epoch,
            ave_loss, err))



def test(cuda, epoch, net, testLoader, optimizer):
    net.eval()
    test_loss = 0
    incorrect = 0
    
    with torch.no_grad():
        for value in testLoader:
            data,target=value['input'].unsqueeze(1), value['label'].squeeze(1)
            if cuda:
                data, target = data.cuda(), target.cuda()
           # ipdb.set_trace()
            
            output = net(data)
            y=F.cross_entropy(output, target)
            test_loss += y.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
           # ipdb.set_trace()
            incorrect += pred.ne(target.view_as(pred)).cpu().sum().item()
             # get the index of the max log-probability
            
    
        test_loss = test_loss
        test_loss /= len(testLoader) # loss function already averages over batch size
        nTotal = len(testLoader.dataset)
        err = 100.*incorrect/nTotal
        print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.4f}%)\n'.format(
            test_loss, incorrect, nTotal, err))
        return err



# def adjust_opt(optAlg, epoch):
#         lr =2e-4
#         if epoch < 40: 
#             lr = 2e-4
#         elif epoch >= 40: 
#             lr = 1e-4
#         elif epoch >= 120: 
#             lr = 1e-4
#         return lr




def main():
    #这里要改回来
    batch=4
    test_err=100
    set_epoch=80
    PATH = './result/learningnet'
    out=open('output.txt','w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = densenet.DenseNet(growthRate=16, depth=40, reduction=0.5,
                            bottleneck=True, nClasses=2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    cuda=  torch.cuda.is_available()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_dataset = be.TrainDataset('train_val.csv',transform=transforms.Compose([be.Getinput(32),be.ToTensor()]));
    test_dataset =be.TrainDataset ('val.csv', transform=transforms.Compose([be.Getinput(32,train=False),be.ToTensor()]))
    train_dataloader= DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=0)
    test_dataloader=DataLoader(test_dataset,batch_size=20,shuffle=True,num_workers=0)
    exam_dataset=be.TrainDataset('exam.csv',train=False,transform=transforms.Compose([be.Getinput(32,train=False),be.ToTensor()]))
    #训练过程
    for epoch in range(1,set_epoch+1):
 #       lr=adjust_opt( optimizer, epoch)
 
        train(cuda,epoch,net,train_dataloader,optimizer,out)
        a=test(cuda, epoch, net, test_dataloader, optimizer)
        
        if test_err > a:
            test_err = a 
            
            torch.save(net.state_dict(), PATH+str(epoch)+'.pth')
            print('saved\n')
            result1=[]
            result2=[]
            name=[]
            for data in exam_dataset:
                value = data['input'].unsqueeze(0)
                value=value.unsqueeze(1).to(device)
                name.append(data['name'])
                with torch.no_grad():
                    output = net(value)
                result1.append(output[0].item())
                result2.append(output[1].item())
            
            array=[]
            array.append(name)
            array1=[]
            array1.append(result1)
            array1.append(result2)
            array1=np.array(array1).T
            result_tensor=torch.from_numpy(array1)
            
            result_tensor=F.softmax(result_tensor,1)
            array.append(list(result_tensor[...,1]))
            array=np.array(array).T
            save = pd.DataFrame(array, columns = ['Id','Predicted']) 
            save.to_csv('./result/exam_result'+str(epoch)+'.csv',index=False)
            another_save=pd.DataFrame(array1,columns=['pro_false','pro_true'])
            another_save.to_csv('./result/original'+str(epoch)+'.csv',index=False)
          #  ipdb.set_trace()
        
        if test_err<18:
            break
    out.close()


if __name__=='__main__':
    main()
