# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 08:23:25 2019

@author: 63195
"""
import densenet as de
import torch
import pandas as pd
import befortraining as be
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

#test.py文件在windows系统下使用
#在GPU可用时使用GPU


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = de.DenseNet(growthRate=16, depth=40, reduction=0.5,
                            bottleneck=True, nClasses=2).to(device)
#
PATH = './learningnet50.pth'
net.eval()
net.load_state_dict(torch.load(PATH))
#通过exam.csv获取测试集的病人编号
exam_dataset=be.TrainDataset('exam.csv',train=False,transform=transforms.Compose([be.Getinput(32,train=False),be.ToTensor()]))
result1=[]
result2=[]
name=[]
array=[]
for data in exam_dataset:
    value = data['input'].unsqueeze(0)
    value=value.unsqueeze(1).to(device)
    name.append(data['name'])
    with torch.no_grad():
        output = net(value)
    result1.append(output[0].item())
    result2.append(output[1].item())
    #判断分数
array.append(name)
array1=[]
array1.append(result1)
array1.append(result2)
array1=np.array(array1).T
result_tensor=torch.from_numpy(array1)
#分数使用softmax函数进行转换

result_tensor=F.softmax(result_tensor,1)
result_numpy=np.array(result_tensor)
array.append(list(result_numpy[...,1]))
array=np.array(array).T
save = pd.DataFrame(array, columns = ['Id','Predicted']) 

save = pd.DataFrame(array, columns = ['Id','Predicted']) 
save.to_csv('submission.csv',index=False)
