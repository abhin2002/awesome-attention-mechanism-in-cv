import torch
import torch.nn as nn
from attention_modules.cbam import ChannelAttention, SpatialAttention  # Import CBAM from your .py file

class VGG11_CBAM(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11_CBAM, self).__init__()

        self.features = nn.Sequential(
            self.conv_block(3, 64),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            self.conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self.conv_block(128, 256),
            self.conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self.conv_block(256, 512),
            self.conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self.conv_block(512, 512),
            self.conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )

        self.classifier = nn.Linear(512, num_classes)

    def conv_block(self, in_channels, out_channels):
        """Convolutional Block with CBAM"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),  # CBAM Channel Attention
            SpatialAttention()  # CBAM Spatial Attention
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.classifier(out)
        return out

# Testing the model
if __name__ == "__main__":
    model = VGG11_CBAM(num_classes=10)
    x = torch.randn(2, 3, 32, 32)  # Batch of 2, 3 color channels, 32x32 image
    y = model(x)
    print(y.shape)  # Expected output: torch.Size([2, 10])
