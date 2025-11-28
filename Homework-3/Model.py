# simple_net_encoder.py
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 
                              stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class SimpleNetEncoder(nn.Module):
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = BottleneckBlock(16, 8, 32, stride=2)
        self.stage2 = BottleneckBlock(32, 16, 64, stride=2)  
        self.stage3 = BottleneckBlock(64, 32, 128, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x