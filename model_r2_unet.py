# we have defined a Unet model for Segmentation

import torch
import torch.nn as nn
from torchsummary import summary

#define smaller blocks:

#To easiliy change activation function etc. for all convolutions
def Conv2d_block(in_ch, out_ch, kernel_size, padding, act_fn=nn.ReLU() ):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        act_fn
    )

#TODO ReLU
# in_ch size = out_ch size
class Rec_block(nn.Module):
    def __init__(self, in_ch, t=2):
        super().__init__()
        self.rec_conv = Conv2d_block(in_ch, in_ch, kernel_size = 3, padding =1)
        self.t = t
        
    def forward(self, x_in):
        for i in range(self.t):
            if i == 0:
                x = self.rec_conv(x_in)
            else:
                x = self.rec_conv(x + x_in)
                #maybe self.rec_conv(x) + x_in
        return x

#RRCNN-block with two recurrent_convolution blocks
class RRCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, t=2):
        super().__init__()
        self.rec_block1 = Rec_block(out_ch)
        self.rec_block2 = Rec_block(out_ch)
        self.forward_conv = Conv2d_block(in_ch, out_ch, kernel_size = 3, padding =1)
        
    def forward(self, x):
        x_f = self.forward_conv(x)
        x1 = self.rec_block1(x_f)
        x2 = self.rec_block2(x1)
        return x2 + x_f

#R2U-Net Model
class R2unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, t=2, num_classes=19, depth=5):
        super().__init__()
        self.depth = depth
        enc_list = []
        for _ in range(self.depth):
            enc_list.append( RRCNN(in_ch=in_ch, out_ch=out_ch) )
            in_ch = out_ch
            out_ch *= 2
        self.enc_list = nn.ModuleList(enc_list)
        
        dec_list = []
        trans_conv_list = []
        out_ch = in_ch // 2
        for _ in range(self.depth-1):
            dec_list.append( RRCNN(in_ch=in_ch, out_ch=out_ch) )
            trans_conv_list.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2 ))
            in_ch = out_ch
            out_ch //= 2
        
        #def as model parameters
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)     
        self.trans_conv_list = nn.ModuleList(trans_conv_list)
        
        # 2x2 max_pool
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv1x1 with Softmax activation
        self.conv1x1 = Conv2d_block(in_ch, num_classes, kernel_size = 1, padding = 0,act_fn= nn.Identity())
    def forward(self, x):
        skip_con = [] #the skip connection between encoding and decoding
        
        #encoding part
        for i in range(self.depth):
            x = self.enc_list[i](x)
            if (i<self.depth-1): # save skip_con and apply max pooling except for last layer
                skip_con.append(x)
                x = self.max_pool(x)
        
        #decoding part
        for i in range(self.depth-1):
            x = self.trans_conv_list[i](x)
            x = self.dec_list[i](x) + skip_con[self.depth-2-i] # decoding + skip_connectionSS
            
        return self.conv1x1(x) # conv1x1 then softmax

 
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        torch.nn.init.constant_(m.bias.data, 0)  