import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """A standard block of two convolutional layers used throughout the U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. ENCODER (Extracting patterns)
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # 2. BOTTLENECK (Deepest feature representation)
        self.bottleneck = DoubleConv(128, 256)
        
        # 3. DECODER (Rebuilding the clean image)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  
        
        # 4. SUPER-RESOLUTION HEAD (Scaling 128x128 up to 256x256)
        self.sr_up = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Run the signal through the Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Run through Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Run through Decoder with Skip Connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        # Final 2x Super-Resolution boost
        out = self.sr_up(d2)
        return out