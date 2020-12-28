from dataset import *
from model import Model
from utils import accuracy_

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import transforms
import matplotlib.pyplot as plt

'''
np.random.seed(987)
torch.manual_seed(987)
if torch.cuda.is_available():
    torch.cuda.manual_seed(987)
'''
def seed_init():
    seed = 892107
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    
    _iter, losses, acc = 0,0,0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        predicts = model(imgs)
        loss = criterion(predicts, labels)
        
        losses += loss.item()
        acc += accuracy_(predicts, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _iter += 1
        print('\t loss:%.3f acc:%.5f'%(losses/_iter, acc/_iter), end='  \r')
    
    print('\t train loss:%.3f acc:%.5f, '%(losses/_iter, acc/_iter))
    return acc/_iter, losses/_iter

@torch.no_grad()
def valid(train_loader, model, criterion, device):
    model.eval()
    
    _iter, losses, acc = 0,0,0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        predicts = model(imgs)
        loss = criterion(predicts, labels)
        
        losses += loss.item()
        acc += accuracy_(predicts, labels)

        _iter += 1
        print('\t loss:%.3f acc:%.5f'%(losses/_iter, acc/_iter), end='  \r')

    print('\t valid loss:%.3f acc:%.5f'%(losses/_iter, acc/_iter))
    return acc/_iter, losses/_iter

if __name__=='__main__':
    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.15,contrast=0.15,saturation=0.15),
        #transforms.RandomAffine(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.ToTensor()])
    # Datasets
    train_set = Dog_Cat_Dataset('./Cat_Dog_data_small/train',112,transform_train )
    valid_set = Dog_Cat_Dataset('./Cat_Dog_data_small/valid',112,transform_test)        

    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=64) #128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_init()
    # Model
    model = Model()
    model.to(device)
    print(model)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss_all = []
    train_acc_all = []

    valid_loss_all = []
    valid_acc_all = []

    best_acc = 0
    for epoch in range(100): #TODO:
        print(f'Epoch {epoch}')
        
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, device)

        valid_acc, valid_loss = valid(valid_loader, model, criterion, device)

        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)

        valid_loss_all.append(valid_loss)
        valid_acc_all.append(valid_acc)
        
        if valid_acc > best_acc:
            print('\t save weights')
            torch.save(model.state_dict(),'best_model.pth')
            best_acc = valid_acc
            
    plt.title("Cross entropy loss learning curve")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.plot(valid_loss_all, label = "validation")
    plt.legend(loc='best')
    plt.plot(train_loss_all, label = "training")
    plt.legend(loc='best')
    plt.savefig("MSE_loss_p2.png")
