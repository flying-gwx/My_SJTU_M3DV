# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 08:23:25 2019

@author: 63195
"""
import densenet as de
import torch
import pandas as pd
import learning as be
import torchvision.transforms as transforms
import numpy as np
import ipdb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = de.DenseNet(growthRate=12, depth=10, reduction=0.5,
                            bottleneck=True, nClasses=2).to(device)

PATH = './learningnet22.pth'
#ipdb.set_trace()
net.load_state_dict(torch.load(PATH))
net.eval()
exam_dataset=be.TrainDataset('exam.csv',train=False,transform=transforms.Compose([be.Getinput(32,train=False),be.ToTensor()]))
result1=[]
result2=[]
name=[]
train_dataset = be.TrainDataset('train_val.csv',transform=transforms.Compose([be.Getinput(32),be.ToTensor()]));
for data in exam_dataset:
    value = data['input'].unsqueeze(0)
    value=value.unsqueeze(1).to(device)
    name.append(data['name'])
    with torch.no_grad():
        output = net(value)
    result1.append(output[0].item())
    result2.append(output[1].item())
    #写进数组中
array=[]
array.append(name)
array.append(result2)
array=np.array(array)
array=array.T
save = pd.DataFrame(array, columns = ['Id','Predicted']) 
save.to_csv('exam_result_22.csv',index=False)
another_save=pd.DataFrame(result1,columns=['predicted'])
another_save.to_csv('another.csv',index=False)