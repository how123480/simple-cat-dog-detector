import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random

def sec_order_transform(X):
    d = 4
    t_X = X
    for i in range(d):
        for j in range(i+1, d):
            t_X = np.concatenate((t_X, (X[:,i]*X[:,j]).reshape(-1,1)),axis = 1).astype(float)
    t_X = np.concatenate((t_X, X**2),axis = 1).astype(float)
    t_X = np.concatenate((np.ones([X.shape[0],1]),t_X),axis = 1).astype(float)
    #print(t_X.shape)
    return t_X

def seed_init():
    seed = 892107
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

df = pd.read_csv('DS_hw6_p1.csv')

# preprocessing
df_x = preprocessing.scale(df.iloc[:,:-2], axis = 0)
train_x = sec_order_transform(df_x[:280, 1:])
#train_x = df_x[:280, 1:]
print("train_x shape: ", train_x.shape)
#test_x = df_x[280:, 1:]
test_x = sec_order_transform(df_x[280:, 1:])
train_y = df.iloc[:280, -2:].to_numpy()

# model structure
class linearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(linearRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(),
            nn.Linear(n_hidden, n_output))

    def forward(self, x):
        out = self.linear(x)
        return out

# training settings
inputDim = 15  #4      # takes variable 'x' 
outputDim = 2       # takes variable 'y'
learningRate = 0.01 
epochs = 100

seed_init()

model = linearRegression(inputDim, 20, outputDim)
print(model) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

# train
loss_all = []
for t in range(100):
    x = Variable(torch.from_numpy(train_x).float())
    y = Variable(torch.from_numpy(train_y).float())
    model.train()
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_func(prediction, y)
    loss.backward()
    loss_all.append(loss.detach().numpy())
    print('epoch = {}, loss = {}'.format(t,loss.detach().numpy()))
    optimizer.step()

# predict
model.eval()
x = Variable(torch.from_numpy(test_x).float())
pred_test = model(x)
print(pred_test)

plt.plot(loss_all)
plt.savefig("MSE_loss.png")