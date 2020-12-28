import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            #nn.Conv2d(16, 16, 1),

            nn.Conv2d(16, 16, 3, 1, 1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(16, 16, 1),

            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(16, 32, 3, 1, 1, groups=16),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 32, 1),

            nn.Conv2d(32, 32, 3, 1, 1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 32, 1),

            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, 1),

            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 64, 3, 1, 1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, 1),

            nn.MaxPool2d(2, 2, 0),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 64, 3, 1, 1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, 1),

            nn.Conv2d(64, 64, 3, 1, 1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, 1),

        )
        #self.avg_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Sequential(
            nn.Linear(64*14*14, 512),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(512, 64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(64, 2),
            nn.ReLU()
        )
        
    def forward(self,x):
        bsize = x.size(0)
        
        x = self.cnn(x)
        #x = self.avg_pool(x)
        x = self.fc(x.view(bsize,-1))
        return x
    
