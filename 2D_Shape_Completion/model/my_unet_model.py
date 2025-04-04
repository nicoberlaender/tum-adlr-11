from model.unet_parts import *
import torch.nn.init as init

class UNet(nn.Module):
    def __init__(self, n_channels, n2_channels, n_classes, bilinear=False, sigmoid = True):
        """Initialize U-Net model.
        Args:
            n_channels (int): Number of input channels
            n2_channels (int): Number of channels in first layer
            n_classes (int): Number of output classes
            bilinear (bool, optional): Use bilinear upsampling. Defaults to False.
            sigmoid (bool, optional): Apply sigmoid activation. Defaults to True.
        """
        
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = sigmoid
        

        self.inc = (DoubleConv(n_channels, n2_channels))
        self.down1 = (Down(n2_channels, n2_channels*2))
        self.down2 = (Down(n2_channels*2, n2_channels*4))
        self.down3 = (Down(n2_channels*4, n2_channels*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(n2_channels*8, n2_channels*16 // factor))
        self.up1 = (Up(n2_channels*16, n2_channels*8 // factor, bilinear))
        self.up2 = (Up(n2_channels*8, n2_channels*4 // factor, bilinear))
        self.up3 = (Up(n2_channels*4, n2_channels*2// factor, bilinear))
        self.up4 = (Up(n2_channels*2, n2_channels, bilinear))
        self.outc = (OutConv(n2_channels, n_classes))
        

        # Initialize the weights of the network
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """
        Initialize weights for Conv2d and BatchNorm2d layers using He initialization.
        Args:
            m: Module to initialize
        """
        
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
        """
        Forward pass of the UNet model.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output logits, with sigmoid activation if specified
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if ( self.sigmoid == True):
            logits= torch.nn.functional.sigmoid(logits)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)