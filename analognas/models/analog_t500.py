import torch
import torch.nn as nn
import torch.nn.functional as F

class IMCBlock(nn.Module):
    """
    IMC-optimized block that considers hardware constraints for analog computing
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(IMCBlock, self).__init__()
        # Use 1x1 convolutions to reduce hardware resource usage
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Depthwise separable convolution for efficiency
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                              padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 projection
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class T500Network(nn.Module):
    """
    AnalogNAS T500 architecture optimized for IMC hardware
    """
    def __init__(self, num_classes=10, in_channels=3):
        super(T500Network, self).__init__()
        # Initial convolution with reduced channels, supporting variable input channels
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        # IMC-optimized feature extraction
        self.layer1 = self._make_layer(8, 16, 2)
        self.layer2 = self._make_layer(16, 32, 2)
        self.layer3 = self._make_layer(32, 64, 2)
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(IMCBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(IMCBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
