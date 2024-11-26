from model.simple_unet_parts import *
import torch.nn.init as init

class UNet(nn.Module):
    def __init__(self, n_channels, n2_channels, n_classes, bilinear=False, sigmoid = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = sigmoid
        

        self.inc = (DoubleConv(n_channels, n2_channels*2))
        self.down1 = (Down(n2_channels*2, n2_channels*4))
        self.down2 = (Down(n2_channels*4, n2_channels*8))
        self.up1 = (Up(n2_channels*8, n2_channels*4, bilinear))
        self.up2 = (Up(n2_channels*4, n2_channels*2, bilinear))
        self.outc = (OutConv(n2_channels*2, n_classes))
        

        # Initialize the weights of the network
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Apply He initialization (Kaiming initialization)
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            # Apply normal initialization for batch normalization layers
            init.ones_(m.weight)
            init.zeros_(m.bias)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3, x2)
        x = self.up2(x4, x1)
        logits = self.outc(x)
        if ( self.sigmoid == True):
            logits= torch.nn.functional.sigmoid(logits)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)