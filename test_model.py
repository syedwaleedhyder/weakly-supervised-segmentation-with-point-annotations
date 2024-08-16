import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(n_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.dec1 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec3 = self.conv_block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d1 = self.dec1(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.final(d3)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model and move it to the appropriate device
model = UNet(n_channels=3, n_classes=24).to(device)

# Loop over powers of two for batch sizes
for i in range(0, 8):  # This will test batch sizes of 2^1 to 2^7 (2 to 128)
    batch_size = 2 ** i
    print(f"Testing batch size: {batch_size}")

    # Create a random tensor to simulate a batch of input images
    x = torch.rand(batch_size, 3, 1500, 1500).to(device)  # Adjust the batch size here

    # Send the input through the model
    output = model(x)
    print(f"Output shape: {output.shape}")