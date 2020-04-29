import torch.nn as nn
import torch
from layer import CapsuleLayer
from torch.autograd import Variable
import torch.nn.functional as F
import config

class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=16, stride=1)
        self.pool=nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=12, stride=1)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=9, stride=1)
        self.relu=nn.ReLU()
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=config.NUM_CLASSES, num_route_nodes=9248, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * config.NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x =F.relu(self.conv3((F.relu(self.conv2(self.pool(F.relu(self.conv1(x), inplace=True)))))))
        x = self.primary_capsules(x)
        
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
         

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(config.NUM_CLASSES)).to(config.DEVICE).index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions