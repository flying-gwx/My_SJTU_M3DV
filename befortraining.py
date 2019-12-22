# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:45:58 2019

@author: 63195
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torchvision import utils
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import ipdb
# load the train_data
'''
title='./train_val/candidate'
final='.npz'
result=title+'1'+final
print(result)
tmp=np.load(result)
'''

'''

这里考虑思路
输入和mask一起用，减小输入
输出
'''

# define the Dataset
class TrainDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.train_val = pd.read_csv(csv_file)
        self.train= train  
        self.transform = transform

    def __len__(self):
        
        return len(self.train_val)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.train:
            patiant_name = self.train_val.iloc[idx,0]
            data = np.load('./train_val/'+patiant_name+'.npz')
            
            label = self.train_val.iloc[idx,1]
            sample = {'patiant_name': patiant_name, 'data': data['voxel'],
                  'mask':data['seg'],'label':label}
            if self.transform:
                sample = self.transform(sample)
        else:
            patiant_name = self.train_val.iloc[idx,0]
            data = np.load('./test/'+patiant_name+'.npz')
            
            sample = {'patiant_name': patiant_name, 'data': data['voxel'],
                  'mask':data['seg']}
            sample=self.transform(sample)
        return sample
    
# define the transform by Getinput and ToTensor
class Getinput(object):

    def __init__(self, one_output,train= True):
        assert isinstance(one_output, (int, tuple))
        self.output_size = one_output**3
        self.one_output=one_output
        self.train= train
#这里需要简单更改
        
    def __call__(self, sample):
        for_list=list(sample.keys())
     #   ipdb.set_trace()
        if len(for_list)>= 4:
            data, mask,label = sample['data'], sample['mask'],sample['label']
        else:
            data,mask= sample['data'],sample['mask']
        delta=data[mask]
        # 病灶大小
        self.size=delta.shape[0]
        self.name=sample['patiant_name']
        result=np.zeros((self.one_output,self.one_output,self.one_output))
        # 这里的判断实际上是将大小变为output_size,不够补零，多了的话均匀取
        if self.output_size >= delta.shape[0]:
            index=np.array(np.where(mask==True))
        #    ipdb.set_trace()
            a=index[...,int(index.shape[1]/2)]
            x,y,z=a[0],a[1],a[2]
            if x-16<0 or x+16>100 or y-16<0 or y+16>100 or z-16<0 or z+16 >100:
                result=data[25:57,25:57,25:57]
            else:
                result=data[x-16:x+16,y-16:y+16,z-16:z+16]
            
        else:
            stride=int(delta.shape[0]/self.output_size)
            delta=delta[0:stride*self.output_size:stride]
            result=delta.reshape(self.one_output,self.one_output,self.one_output) 
        a=torch.rand(1).item()
        if a> 0.5:
            result=np.rot90(result,1,(1,2))
        elif a < 0.9:
            result=np.flip(result,1)
        else:
            result=result
            

        if self.train:
            return {'data': result, 'label': label,'size':self.size}
        else:
            return {'data': result, 'size':self.size,'name':self.name}
            
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        b=list(sample.keys())
        if b[1] == 'label':
            label=np.array([sample['label']])
            return {'input': torch.from_numpy(sample['data']/255).type(torch.FloatTensor),
                'label': torch.from_numpy(label).type(torch.LongTensor),
                'size':sample['size']}
        else :
            return  {'input': torch.from_numpy(sample['data']/255).type(torch.FloatTensor),
                     'name':sample['name'],
                     'size': sample['size']}


############################################
            #开始主程序
            #要训练网络
        #如果一个简单的Lnnet能不能出结果

#x_train,x_label,y_train,y_label=train_test_split(train_dataset[])

#########################################
#这里是训练过程
#包括网络设计和train以及test的设计
def train( model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss=0
    for batch_idx, data in enumerate(train_loader):
        value, label = data['input'].unsqueeze(1).to(device), data['label'].to(device)
        optimizer.zero_grad()
      #  if data.dim() == 3:
       #     data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)
   #     ipdb.set_trace()
        output = model(value)
        
        loss=F.l1_loss(output, label)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
    train_loss/=len(train_loader.dataset)
    print('\nTrain set: {:.4f}'.format(
        train_loss))
    print('==================\n')
    return train_loss
def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    i = 0
    
    with torch.no_grad():
        for data in test_loader:
            value,label = data['input'].unsqueeze(1).to(device), data['label'].to(device)
            
            output = model(value)
            i=i+1
            
            test_loss += F.l1_loss(output, label).item() # sum up batch loss
            #ipdb.set_trace()
            

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss))
    return test_loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, kernel_size=3,stride=1)
        self.pool = nn.MaxPool3d(3, stride=2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=3)
        self.conv3= nn.Conv3d(16, 64, kernel_size=1)
        self.fc1 = nn.Linear(64 * 5 * 5*5, 120)
        self.fc2 = nn.Linear(120, 42)
        self.fc3 = nn.Linear(42, 1)

    def forward(self, x):
        y=self.conv1(x)
        y=F.relu(y)
        y = self.pool(y)
        
      #  ipdb.set_trace()
        
        
        #(6,14,14,14)
        # 卷积后变成（16,12,12,12）
        y=F.relu(self.conv3(F.relu(self.conv2(y))))
        y = self.pool(y)
       # 池化后变成(16,5,5,5)
        y = y.view(-1, 64 * 5 * 5 * 5 )
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        
        y = self.fc3(y)
        return y
if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=Net().to(device)
    
    train_dataset = TrainDataset('train_val.csv',transform=transforms.Compose([Getinput(32),ToTensor()]));
    batch=100
    test_dataset =TrainDataset ('val.csv', transform=transforms.Compose([Getinput(32),ToTensor()]))
    train_dataloader= DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=0)
    test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=True,num_workers=0)
    exam_dataset=TrainDataset('exam.csv',train=False,transform=transforms.Compose([Getinput(32,train=False),ToTensor()]))
    #exam_dataloader=DataLoader(exam_dataset,batch_size=1,shuffle=False,num_workers=0)
    
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_last=1
    for epoch in range(80):
        train_loss=train(net,device,train_dataloader,optimizer,epoch)
        if epoch > 60:
            
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        LOSS= test(net,device,test_dataloader)
        if LOSS<0.4:
            break;
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    '''
    result=[]
    name=[]
    for data in exam_dataset:
        value = data['input'].unsqueeze(0)
        value=value.unsqueeze(1).to(device)
        name.append(data['name'])
        output = net(value)
        result.append(output.item())
    array=[]
    array.append(name)
    array.append(result)
    array=np.array(array)
    array=array.T
    save = pd.DataFrame(array, columns = ['Id','Predicted']) 
    save.to_csv('exam_result.csv',index=False)
  '''   
            
    
    
'''
这里是一个实验，看dataset写好了没
size_of_task=[]
labels=[]
for i in range(len(train_dataset)):
    size_of_task.append(train_dataset[i]['size']) 
    labels.append(train_dataset[i]['label'])
data_array=[]
data_array.append(size_of_task)
data_array.append(labels)

np_array=np.array(data_array)
np_array=np_array.T
save = pd.DataFrame(np_array, columns = ['size', 'label']) 
save.to_csv('size_label.csv',index=False,header=False)
'''

  

