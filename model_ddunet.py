# we have defined a Unet model for Segmentation

import torch
import torch.nn as nn
from torchsummary import summary

def conv_block_bn(in_channels, out_channels,dil):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1,dilation=dil),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1,dilation=dil),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels)
    )

class M_Unet(nn.Module):
  
    def __init__(self,n_class):
        super(M_Unet, self).__init__()
        #define the layers for your model
        self.conv_down1 = conv_block_bn(3, 64,1)
        self.conv_down2 = conv_block_bn(64, 128,1)
        self.conv_down3 = conv_block_bn(128, 256,3)
        self.conv_down4 = conv_block_bn(256, 512,3)        

        self.maxpool = nn.MaxPool2d(2)

        self.conv_transpose3 = nn.ConvTranspose2d(512,512,kernel_size=3,stride=3,dilation=1,padding=2)        
        self.conv_transpose2 = nn.ConvTranspose2d(256,256,kernel_size=3,stride=3,dilation=1,padding=8)
        self.conv_transpose1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2,dilation=1,padding=0)

        self.conv_up3 = conv_block_bn(256 + 512, 256,3)
        self.conv_up2 = conv_block_bn(128 + 256, 128,1)
        self.conv_up1 = conv_block_bn(128 + 64, 64,1)
        
        self.conv_last = nn.Conv2d(64,out_channels= n_class,kernel_size= 1)
    

    def forward(self, x):
        #define the forward pass
        conv_1 = self.conv_down1(x)
        conv_1_mp = self.maxpool(conv_1)

        conv_2 = self.conv_down2(conv_1_mp)
        conv_2_mp = self.maxpool(conv_2)

        conv_3 = self.conv_down3(conv_2_mp)
        conv_3_mp = self.maxpool(conv_3)     

        conv_4 = self.conv_down4(conv_3_mp)

        up_3 = self.conv_transpose3(conv_4)

        up_cat_3 = torch.cat([up_3, conv_3], dim=1)

        up_conv_3 = self.conv_up3(up_cat_3)

        up_2 = self.conv_transpose2(up_conv_3)

        up_cat_2 = torch.cat([up_2, conv_2], dim=1)

        up_conv_2 = self.conv_up2(up_cat_2)
        

        up_1 = self.conv_transpose1(up_conv_2)

        up_cat_1 = torch.cat([up_1, conv_1], dim=1)


        up_conv_1 = self.conv_up1(up_cat_1)  
        out = self.conv_last(up_conv_1)
        return out
 
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        torch.nn.init.constant_(m.bias.data, 0)  