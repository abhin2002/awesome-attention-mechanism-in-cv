import torch
import torch.nn as nn
from attention_modules.TripletAttention import TripletAttention  # Make sure to use your .py file correctly

class VGG11_TripletAttention(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11_TripletAttention, self).__init__()

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
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(512, num_classes)

    def conv_block(self, in_channels, out_channels):
        """VGG-style convolution block with Triplet Attention"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            TripletAttention(gate_channels=out_channels)  # Plug in Triplet Attention here
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# ✅ Test model structure
if __name__ == "__main__":
    model = VGG11_TripletAttention(num_classes=10)
    x = torch.randn(2, 3, 32, 32)  # For CIFAR-10
    y = model(x)
    print(y.shape)  # Should print: torch.Size([2, 10])
