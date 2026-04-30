from math import sqrt
import torch
import torch.nn as nn
import os.path
import warnings
import requests
from tqdm import tqdm
try:
    from slumworldML.src.registry import MODELS_REGISTRY
except ImportError:
    try:
        from src.model import MODELS_REGISTRY
    except ImportError:
        import sys
        sys.path.append("/home/yuting/data/slumworld")
        from slumworldML.src.model import MODELS_REGISTRY
from torchvision import models
from transformers import Swinv2Model, SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerImageProcessor, SegformerConfig
import torchvision.transforms.functional as TF
import torch.nn.functional as F
try:
    from slumworldML.src.utilities import double_conv, get_vgg_encoder_stages
    from slumworldML.src.pvtv2 import pvt_v2_b2
except Exception as Err1:
    try:
        from utilities import double_conv, get_vgg_encoder_stages
    except Exception as Err2:
        from src.utilities import double_conv, get_vgg_encoder_stages
from torch.autograd import Function

class GradientReversal(Function):
    
    @staticmethod
    def forward(ctx, x):
        return x
    
    def backward(ctx, grad_output):
        return grad_output.neg()

def reverse_gradients(x):
    return GradientReversal.apply(x)



class FCNHead(nn.Sequential):
    '''Basic Classification head for domain adaptation'''
    def __init__(self, in_channels: int, out_channels: int, pooldim: int = 1, dim_reduction=8) -> None:
        inter_channels_1 = in_channels // dim_reduction
        inter_channels_2 = inter_channels_1 // dim_reduction
        layers = [
            nn.AdaptiveAvgPool2d(pooldim),
            nn.Flatten(),
            nn.Linear(in_channels*(pooldim**2), inter_channels_1),
            nn.BatchNorm1d(inter_channels_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_channels_1, inter_channels_2),
            nn.BatchNorm1d(inter_channels_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_channels_2, out_channels),
        ]
        super().__init__(*layers)

# Vanilla UNet
class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels=1, features=[64, 128, 256, 512], 
                 dropout_prob=0.05, dropout_2d_prob=0.05, train_head_only=True, domain_classifier=True):
        super().__init__()
        self.name = "UNet"
        self.encoding_parts = nn.ModuleList()
        self.decoding_parts = nn.ModuleList()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.use_dropout = dropout_prob > 0.0
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout_2d = dropout_2d_prob > 0.0
        self.dropout_2d = nn.Dropout2d(p=dropout_2d_prob)
                
        for feature in features:
            self.encoding_parts.append(double_conv(in_channels, feature))
            in_channels = feature
        # self.encoding_parts.apply(init_conv2d)
        self.middle_part = double_conv(features[-1], features[-1]*2)
        
        for feature in reversed(features):
            self.decoding_parts.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2, padding=0, output_padding=0))
            self.decoding_parts.append(double_conv(feature*2, feature))
        # self.decoding_parts.apply(init_conv2d)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=features[-1]*2, out_channels=out_channels, pooldim=1)
    
    def forward(self, x):
        
        skip_connections = [] # Store skip connections in order to be able to concatenate later

        # Encoding part        
        for enc_part in self.encoding_parts:
            x = enc_part(x) # Double convolution
            skip_connections.append(x)
            x = self.max_pool(x)
            if self.use_dropout:
                x = self.dropout(x)
            if self.use_dropout_2d:
                x = self.dropout_2d(x)

        # Middle part
        middle_part = self.middle_part(x) 
        
        skip_connections = skip_connections[::-1] # Reverse skip connections for concatenation in the decoding part
        
        # Decoding part 
        # get the first filter manually in order to be able to access the middel part as a separate variable (for the aux classifier)
        x = self.decoding_parts[0](middle_part)
        skip_connection = skip_connections[0]
        if x.shape != skip_connection.shape: # Address issue of odd pixel number of width and height
            x = TF.resize(x, size=skip_connection.shape[2:])
        
        x = torch.cat((x, skip_connection), dim=1) # Apply skip connection

        if self.use_dropout:
            x = self.dropout(x)
        if self.use_dropout_2d:
            x = self.dropout_2d(x)

        x = self.decoding_parts[1](x) # Double convolution
            
        for i in range(2, len(self.decoding_parts), 2):
            x = self.decoding_parts[i](x) # Transposed convolution

            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape: # Address issue of odd pixel number of width and height
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            x = torch.cat((x, skip_connection), dim=1) # Apply skip connection

            if self.use_dropout:
                x = self.dropout(x)
            if self.use_dropout_2d:
                x = self.dropout_2d(x)

            x = self.decoding_parts[i+1](x) # Double convolution
            
        final_conv = self.final_conv(x)

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with VGG11 encoder (no batch norm)
class UNet_VGG11(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()

        vgg11_model = models.vgg11(weights=models.VGG11_BN_Weights.IMAGENET1K_V1, progress=True)
    
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = vgg11_model.features
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        if (self.in_channels != 3):
            self.conv0 = double_conv(self.in_channels, 3)

        if train_head_only:
            for parameter in vgg11_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in vgg11_model.parameters():
                parameter.requires_grad = True
        
        self.relu = self.encoder[1]

        self.conv1 = self.encoder[0]
        self.conv1d = self.encoder[3]
        self.conv2 = self.encoder[6]
        self.conv2d = self.encoder[8]
        self.conv3 = self.encoder[11]
        self.conv3d = self.encoder[13]
        self.conv4 = self.encoder[16]
        self.conv4d = self.encoder[18]
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(768, 256)
        
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(384, 128)
        
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(192, 32)
        
        self.final_conv = nn.Conv2d(32, self.out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)
    
    def forward(self, x):
        
        if (self.in_channels != 3):
            conv0 = self.conv0(x)
        else:
            conv0 = x
        
        # Encoding part
        conv1 = self.relu(self.conv1(conv0))
        conv1d = self.relu(self.conv1d(conv1))
        conv1d_m = self.max_pool(conv1d)
        
        conv2 = self.relu(self.conv2(conv1d_m))
        conv2d = self.relu(self.conv2d(conv2))
        conv2d_m = self.max_pool(conv2d)
        
        conv3 = self.relu(self.conv3(conv2d_m))
        conv3d = self.relu(self.conv3d(conv3))
        conv3d_m = self.max_pool(conv3d)
        
        conv4 = self.relu(self.conv4(conv3d_m))
        conv4d = self.relu(self.conv4d(conv4))
        conv4d_m = self.max_pool(conv4d)
        
        # Middle part
        middle_part = self.middle_part(conv4d_m)
        
        # Decoding part
        dec1_transp = self.dec1_transp(middle_part)
        if dec1_transp.shape != conv4d.shape:
            dec1_transp = TF.resize(dec1_transp, size=conv4d.shape[2:])
            
        dec1_cat = torch.cat([conv4d, dec1_transp], 1)
        dec1_double = self.dec1_double(dec1_cat)
        
        dec2_transp = self.dec2_transp(dec1_double)
        if dec2_transp.shape != conv3d.shape:
            dec2_transp = TF.resize(dec2_transp, size=conv3d.shape[2:])
        
        dec2_cat = torch.cat([conv3d, dec2_transp], 1)
        dec2_double = self.dec2_double(dec2_cat)
        
        dec3_transp = self.dec3_transp(dec2_double)
        
        if dec3_transp.shape != conv2d.shape:
            dec3_transp = TF.resize(dec3_transp, size=conv2d.shape[2:])
        
        dec3_cat = torch.cat([conv2d, dec3_transp], 1)
        dec3_double = self.dec3_double(dec3_cat)
        
        dec4_transp = self.dec4_transp(dec3_double)
        
        if dec4_transp.shape != conv1d.shape:
            dec4_transp = TF.resize(dec4_transp, size=conv1d.shape[2:])
        
        dec4_cat = torch.cat([conv1d, dec4_transp], 1)
        dec4_double = self.dec4_double(dec4_cat)
        
        final_conv = self.final_conv(dec4_double)

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with VGG11 encoder and batch normalization
class UNet_VGG11_BN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()
        
        vgg11_bn_model = models.vgg11_bn(weights=models.AnyStageVGG11_BN_Weights.IMAGENET1K_V1, progress=True)
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.encoder = vgg11_bn_model.features
    
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        if (self.in_channels != 3):
            self.conv0 = double_conv(self.in_channels, 3)

        if train_head_only:
            for parameter in vgg11_bn_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in vgg11_bn_model.parameters():
                parameter.requires_grad = True
        
        self.relu = self.encoder[2]

        self.conv1 = self.encoder[0]
        self.conv1bn = self.encoder[1]
        self.conv1d = self.encoder[4]
        self.conv1dbn = self.encoder[5]
        
        self.conv2 = self.encoder[8]
        self.conv2bn = self.encoder[9]
        self.conv2d = self.encoder[11]
        self.conv2dbn = self.encoder[12]
        
        self.conv3 = self.encoder[15]
        self.conv3bn = self.encoder[16]
        self.conv3d = self.encoder[18]
        self.conv3dbn = self.encoder[19]
        
        self.conv4 = self.encoder[22]
        self.conv4bn = self.encoder[23]
        self.conv4d = self.encoder[25]
        self.conv4dbn = self.encoder[26]
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(768, 256)
        
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(384, 128)
        
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(192, 32)
        
        self.final_conv = nn.Conv2d(32, self.out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)
    
    def forward(self, x):
        
        if (self.in_channels != 3):
            conv0 = self.conv0(x)
        else:
            conv0 = x
        
        # Encoding part
        conv1 = self.conv1(conv0)
        conv1bn = self.conv1bn(conv1)
        conv1r = self.relu(conv1bn)
        conv1d = self.conv1d(conv1r)
        conv1dbn = self.conv1dbn(conv1d)
        conv1dr = self.relu(conv1dbn)
        conv1m = self.max_pool(conv1dr)

        conv2 = self.conv2(conv1m)
        conv2bn = self.conv2bn(conv2)
        conv2r = self.relu(conv2bn)
        conv2d = self.conv2d(conv2r)
        conv2dbn = self.conv2dbn(conv2d)
        conv2dr = self.relu(conv2dbn)
        conv2m = self.max_pool(conv2dr)

        conv3 = self.conv3(conv2m)
        conv3bn = self.conv3bn(conv3)
        conv3r = self.relu(conv3bn)
        conv3d = self.conv3d(conv3r)
        conv3dbn = self.conv3dbn(conv3d)
        conv3dr = self.relu(conv3dbn)
        conv3m = self.max_pool(conv3dr)

        conv4 = self.conv4(conv3m)
        conv4bn = self.conv4bn(conv4)
        conv4r = self.relu(conv4bn)
        conv4d = self.conv4d(conv4r)
        conv4dbn = self.conv4dbn(conv4d)
        conv4dr = self.relu(conv4dbn)
        conv4m = self.max_pool(conv4dr)

        # Middle part
        middle_part = self.middle_part(conv4m)
        
        # Decoding part
        dec1_transp = self.dec1_transp(middle_part)
        if dec1_transp.shape != conv4dr.shape:
            dec1_transp = TF.resize(dec1_transp, size=conv4dr.shape[2:])
            
        dec1_cat = torch.cat([conv4dr, dec1_transp], 1)
        dec1_double = self.dec1_double(dec1_cat)
        
        dec2_transp = self.dec2_transp(dec1_double)
        if dec2_transp.shape != conv3dr.shape:
            dec2_transp = TF.resize(dec2_transp, size=conv3dr.shape[2:])
        
        dec2_cat = torch.cat([conv3dr, dec2_transp], 1)
        dec2_double = self.dec2_double(dec2_cat)
        
        dec3_transp = self.dec3_transp(dec2_double)
        if dec3_transp.shape != conv2dr.shape:
            dec3_transp = TF.resize(dec3_transp, size=conv2dr.shape[2:])
        
        dec3_cat = torch.cat([conv2dr, dec3_transp], 1)
        dec3_double = self.dec3_double(dec3_cat)
        
        dec4_transp = self.dec4_transp(dec3_double)
        if dec4_transp.shape != conv1dr.shape:
            dec4_transp = TF.resize(dec4_transp, size=conv1dr.shape[2:])
        
        dec4_cat = torch.cat([conv1dr, dec4_transp], 1)
        dec4_double = self.dec4_double(dec4_cat)
        
        final_conv = self.final_conv(dec4_double)

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with VGG11 encoder and batch normalization. Different implementation.
class UNet_VGG11_BN2(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1,  train_head_only=True, _depth=5, domain_classifier=True):
        super().__init__()

        self.name = 'unet_vgg11_bn'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._depth = _depth

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        vgg11_bn_model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1, progress=True)
        self.encoder = vgg11_bn_model.features

        # TODO: Validate that the new changes with parameter.grad etc work
        self._set_trainable(train_head_only)


        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.middle_part = double_conv(512, 1024)
        
        # self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1536, 512)
        
        # self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(768, 256)
        
        # self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(384, 128)
        
        # self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(192, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)

    def _set_trainable(self, train_head_only):
        self.train_head_only = train_head_only
        if train_head_only:
            for _, parameter in self.encoder.named_parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        else:
            for _, parameter in self.encoder.named_parameters():
                parameter.requires_grad_(True)
                parameter.grad = torch.zeros_like(parameter)

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.encoder:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x):

        stages = self.get_stages()
        features = []
        
        # Pass through Encoder
        x = self.input_adapter(x)
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        # Extract features
        conv1dr = features[0]
        conv2dr = features[1]
        conv3dr = features[2]
        conv4dr = features[3]
        conv4m = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(conv4dr)

        # Middle part
        middle_part = self.middle_part(conv4m)

        # Decoding part
        # dec1_transp = self.dec1_transp(middle_part)
        dec1_transp = F.interpolate(middle_part, scale_factor=2, mode="nearest")

        if dec1_transp.shape != conv4dr.shape:
            dec1_transp = TF.resize(dec1_transp, size=conv4dr.shape[2:])
        
        dec1_cat = torch.cat([conv4dr, dec1_transp], 1)
        dec1_double = self.dec1_double(dec1_cat)
        
        # dec2_transp = self.dec2_transp(dec1_double)
        dec2_transp = F.interpolate(dec1_double, scale_factor=2, mode="nearest")
        if dec2_transp.shape != conv3dr.shape:
            dec2_transp = TF.resize(dec2_transp, size=conv3dr.shape[2:])
        
        dec2_cat = torch.cat([conv3dr, dec2_transp], 1)
        dec2_double = self.dec2_double(dec2_cat)
        
        # dec3_transp = self.dec3_transp(dec2_double)
        dec3_transp = F.interpolate(dec2_double, scale_factor=2, mode="nearest")
        if dec3_transp.shape != conv2dr.shape:
            dec3_transp = TF.resize(dec3_transp, size=conv2dr.shape[2:])
        
        dec3_cat = torch.cat([conv2dr, dec3_transp], 1)
        dec3_double = self.dec3_double(dec3_cat)
        
        # dec4_transp = self.dec4_transp(dec3_double)
        dec4_transp = F.interpolate(dec3_double, scale_factor=2, mode="nearest")
        if dec4_transp.shape != conv1dr.shape:
            dec4_transp = TF.resize(dec4_transp, size=conv1dr.shape[2:])
        
        dec4_cat = torch.cat([conv1dr, dec4_transp], 1)
        dec4_double = self.dec4_double(dec4_cat)
        
        final_conv = self.final_conv(dec4_double)

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with VGG13 encoder and batch normalization
class UNet_VGG13_BN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        vgg13_bn_model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1, progress=True)   
        self.encoder = vgg13_bn_model.features

        if train_head_only:
            for parameter in vgg13_bn_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in vgg13_bn_model.parameters():
                parameter.requires_grad = True

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        self.encoder_blocks = get_vgg_encoder_stages(self.encoder)
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(512, 256)
        
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(256, 128)
        
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)

    def forward(self, x):

        # Encoding part
        x = self.input_adapter(x)

        conv1 = self.encoder_blocks[0](x) # dims: tile * tile
        conv2 = self.encoder_blocks[1](conv1) # dims: tile/2 * tile/2
        conv3 = self.encoder_blocks[2](conv2) # dims: tile/4 * tile/4
        conv4 = self.encoder_blocks[3](conv3) # dims: tile/8 * tile/8
        conv4m = self.max_pool(self.encoder_blocks[4](conv4)) # dims: tile/32 * tile/32

        # Middle part
        middle_part = self.middle_part(conv4m) # dims: tile/32 * tile/32

        # Decoding part
        dec1_transp = self.dec1_transp(middle_part) # dims: tile/16 * tile/16
        if dec1_transp.shape[2:] != conv4.shape[2:]:
            dec1_transp = TF.resize(dec1_transp, size=conv4.shape[2:]) # dims: tile/8 * tile/8
        dec1_cat = torch.cat([conv4, dec1_transp], 1) # dims: tile/8 * tile/8
        dec1_double = self.dec1_double(dec1_cat) # dims: tile/8 * tile/8

        dec2_transp = self.dec2_transp(dec1_double) # dims: tile/4 * tile/4
        if dec2_transp.shape[2:] != conv3.shape[2:]:
            dec2_transp = TF.resize(dec2_transp, size=conv3.shape[2:]) # dims: tile/4 * tile/4
        dec2_cat = torch.cat([conv3, dec2_transp], 1) # dims: tile/4 * tile/4
        dec2_double = self.dec2_double(dec2_cat) # dims: tile/4 * tile/4

        dec3_transp = self.dec3_transp(dec2_double) # dims: tile/2 * tile/2
        if dec3_transp.shape[2:] != conv2.shape[2:]:
            dec3_transp = TF.resize(dec3_transp, size=conv2.shape[2:]) # dims: tile/2 * tile/2
        dec3_cat = torch.cat([conv2, dec3_transp], 1) # dims: tile/2 * tile/2
        dec3_double = self.dec3_double(dec3_cat) # dims: tile/2 * tile/2

        dec4_transp = self.dec4_transp(dec3_double) # dims: tile * tile
        if dec4_transp.shape[2:] != conv1.shape[2:]:
            dec4_transp = TF.resize(dec4_transp, size=conv1.shape[2:]) # dims: tile * tile
        dec4_cat = torch.cat([conv1, dec4_transp], 1) # dims: tile * tile
        dec4_double = self.dec4_double(dec4_cat) # dims: tile * tile

        final_conv =  self.final_conv(dec4_double) # dims: tile * tile

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with VGG16 encoder and batch normalization
class UNet_VGG16_BN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        vgg16_bn_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1, progress=True)   
        self.encoder = vgg16_bn_model.features

        if train_head_only:
            for parameter in vgg16_bn_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in vgg16_bn_model.parameters():
                parameter.requires_grad = True

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        self.encoder_blocks = get_vgg_encoder_stages(self.encoder)
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(512, 256)
        
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(256, 128)
        
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)

    def forward(self, x):

        # Encoding part
        x = self.input_adapter(x)

        conv1 = self.encoder_blocks[0](x) # dims: tile * tile
        conv2 = self.encoder_blocks[1](conv1) # dims: tile/2 * tile/2
        conv3 = self.encoder_blocks[2](conv2) # dims: tile/4 * tile/4
        conv4 = self.encoder_blocks[3](conv3) # dims: tile/8 * tile/8
        conv4m = self.max_pool(self.encoder_blocks[4](conv4)) # dims: tile/32 * tile/32

        # Middle part
        middle_part = self.middle_part(conv4m) # dims: tile/32 * tile/32

        # Decoding part
        dec1_transp = self.dec1_transp(middle_part) # dims: tile/16 * tile/16
        if dec1_transp.shape[2:] != conv4.shape[2:]:
            dec1_transp = TF.resize(dec1_transp, size=conv4.shape[2:]) # dims: tile/8 * tile/8
        dec1_cat = torch.cat([conv4, dec1_transp], 1) # dims: tile/8 * tile/8
        dec1_double = self.dec1_double(dec1_cat) # dims: tile/8 * tile/8

        dec2_transp = self.dec2_transp(dec1_double) # dims: tile/4 * tile/4
        if dec2_transp.shape[2:] != conv3.shape[2:]:
            dec2_transp = TF.resize(dec2_transp, size=conv3.shape[2:]) # dims: tile/4 * tile/4
        dec2_cat = torch.cat([conv3, dec2_transp], 1) # dims: tile/4 * tile/4
        dec2_double = self.dec2_double(dec2_cat) # dims: tile/4 * tile/4

        dec3_transp = self.dec3_transp(dec2_double) # dims: tile/2 * tile/2
        if dec3_transp.shape[2:] != conv2.shape[2:]:
            dec3_transp = TF.resize(dec3_transp, size=conv2.shape[2:]) # dims: tile/2 * tile/2
        dec3_cat = torch.cat([conv2, dec3_transp], 1) # dims: tile/2 * tile/2
        dec3_double = self.dec3_double(dec3_cat) # dims: tile/2 * tile/2

        dec4_transp = self.dec4_transp(dec3_double) # dims: tile * tile
        if dec4_transp.shape[2:] != conv1.shape[2:]:
            dec4_transp = TF.resize(dec4_transp, size=conv1.shape[2:]) # dims: tile * tile
        dec4_cat = torch.cat([conv1, dec4_transp], 1) # dims: tile * tile
        dec4_double = self.dec4_double(dec4_cat) # dims: tile * tile

        final_conv =  self.final_conv(dec4_double) # dims: tile * tile

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with VGG19 encoder and batch normalization
class UNet_VGG19_BN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        vgg19_bn_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1, progress=True)   
        self.encoder = vgg19_bn_model.features

        if train_head_only:
            for parameter in vgg19_bn_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in vgg19_bn_model.parameters():
                parameter.requires_grad = True

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        self.encoder_blocks = get_vgg_encoder_stages(self.encoder)
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(512, 256)
        
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(256, 128)
        
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)

    def forward(self, x):

        # Encoding part
        x = self.input_adapter(x)

        conv1 = self.encoder_blocks[0](x) # dims: tile * tile
        conv2 = self.encoder_blocks[1](conv1) # dims: tile/2 * tile/2
        conv3 = self.encoder_blocks[2](conv2) # dims: tile/4 * tile/4
        conv4 = self.encoder_blocks[3](conv3) # dims: tile/8 * tile/8
        conv4m = self.max_pool(self.encoder_blocks[4](conv4)) # dims: tile/32 * tile/32

        # Middle part
        middle_part = self.middle_part(conv4m) # dims: tile/32 * tile/32

        # Decoding part
        dec1_transp = self.dec1_transp(middle_part) # dims: tile/16 * tile/16
        if dec1_transp.shape[2:] != conv4.shape[2:]:
            dec1_transp = TF.resize(dec1_transp, size=conv4.shape[2:]) # dims: tile/8 * tile/8
        dec1_cat = torch.cat([conv4, dec1_transp], 1) # dims: tile/8 * tile/8
        dec1_double = self.dec1_double(dec1_cat) # dims: tile/8 * tile/8

        dec2_transp = self.dec2_transp(dec1_double) # dims: tile/4 * tile/4
        if dec2_transp.shape[2:] != conv3.shape[2:]:
            dec2_transp = TF.resize(dec2_transp, size=conv3.shape[2:]) # dims: tile/4 * tile/4
        dec2_cat = torch.cat([conv3, dec2_transp], 1) # dims: tile/4 * tile/4
        dec2_double = self.dec2_double(dec2_cat) # dims: tile/4 * tile/4

        dec3_transp = self.dec3_transp(dec2_double) # dims: tile/2 * tile/2
        if dec3_transp.shape[2:] != conv2.shape[2:]:
            dec3_transp = TF.resize(dec3_transp, size=conv2.shape[2:]) # dims: tile/2 * tile/2
        dec3_cat = torch.cat([conv2, dec3_transp], 1) # dims: tile/2 * tile/2
        dec3_double = self.dec3_double(dec3_cat) # dims: tile/2 * tile/2

        dec4_transp = self.dec4_transp(dec3_double) # dims: tile * tile
        if dec4_transp.shape[2:] != conv1.shape[2:]:
            dec4_transp = TF.resize(dec4_transp, size=conv1.shape[2:]) # dims: tile * tile
        dec4_cat = torch.cat([conv1, dec4_transp], 1) # dims: tile * tile
        dec4_double = self.dec4_double(dec4_cat) # dims: tile * tile

        final_conv =  self.final_conv(dec4_double) # dims: tile * tile

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

# UNet with ResNet18 encoder
class UNet_Resnet18(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        resnet18_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1,progress=True)   
        modules = list(resnet18_model.children())

        if train_head_only:
            for parameter in resnet18_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in resnet18_model.parameters():
                parameter.requires_grad = True

        self.relu = nn.ReLU(inplace=True)

        self.encoder_blocks = nn.ModuleList(self.get_stages(modules))
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(512, 256)
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(256, 128)
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(128, 64)
        self.dec5_transp = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec5_double = double_conv(128, 64)
        self.dec6_transp = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec6_double = double_conv(32, 32)
        
        self.final_conv = nn.Conv2d(32, self.out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)
    
    def get_stages(self, modules):
        return [
            nn.Sequential(*modules[:3]),
            nn.Sequential(*modules[3:5]),
            modules[5],
            modules[6],
            modules[7]
        ]

    def forward(self, x):

        # Encoding part
        x = self.input_adapter(x) # dims: tile * tile

        layer0 = self.encoder_blocks[0](x) # dims: tile/2 * tile/2
        layer1 = self.encoder_blocks[1](layer0) # dims: tile/4 * tile/4
        layer2 = self.encoder_blocks[2](layer1) # dims: tile/8 * tile/8
        layer3 = self.encoder_blocks[3](layer2) # dims: tile/16 * tile/16
        layer4 = self.encoder_blocks[4](layer3) # dims: tile/32 * tile/32

        # Middle part
        middle_part = self.middle_part(layer4) # dims: tile/32 * tile/32

        # Decoding part
        dec1_transp = self.dec1_transp(middle_part) # dims: tile/16 * tile/16
        if dec1_transp.shape[2:] != layer4.shape[2:]:
            dec1_transp = TF.resize(dec1_transp, size=layer4.shape[2:])  # dims: tile/32 * tile/32
        dec1_cat = torch.cat([layer4, dec1_transp], 1) # dims: tile/32 * tile/32
        dec1_double = self.dec1_double(dec1_cat) # dims: tile/32 * tile/32

        dec2_transp = self.dec2_transp(dec1_double) # dims: tile/16 * tile/16
        if dec2_transp.shape[2:] != layer3.shape[2:]: # dims: tile/16 * tile/16
            dec2_transp = TF.resize(dec2_transp, size=layer3.shape[2:])
        dec2_cat = torch.cat([layer3, dec2_transp], 1) # dims: tile/16 * tile/16
        dec2_double = self.dec2_double(dec2_cat) # dims: tile/16 * tile/16

        dec3_transp = self.dec3_transp(dec2_double)
        if dec3_transp.shape[2:] != layer2.shape[2:]:
            dec3_transp = TF.resize(dec3_transp, size=layer2.shape[2:])
        dec3_cat = torch.cat([layer2, dec3_transp], 1) # dims: tile/8 * tile/8
        dec3_double = self.dec3_double(dec3_cat) # dims: tile/8 * tile/8

        dec4_transp = self.dec4_transp(dec3_double) # dims: tile/4 * tile/4
        if dec4_transp.shape[2:] != layer1.shape[2:]: # dims: tile/4 * tile/4
            dec4_transp = TF.resize(dec4_transp, size=layer1.shape[2:])
        dec4_cat = torch.cat([layer1, dec4_transp], 1) # dims: tile/4 * tile/4
        dec4_double = self.dec4_double(dec4_cat) # dims: tile/4 * tile/4

        dec5_transp = self.dec5_transp(dec4_double) # dims: tile/2 * tile/2
        if dec5_transp.shape[2:] != layer0.shape[2:]: # dims: tile/2 * tile/2
            dec5_transp = TF.resize(dec5_transp, size=layer0.shape[2:])
        dec5_cat = torch.cat([layer0, dec5_transp], 1) # dims: tile/2 * tile/2
        dec5_double = self.dec5_double(dec5_cat) # dims: tile/2 * tile/2

        dec6_transp = self.dec6_transp(dec5_double) # dims: tile * tile
        if dec6_transp.shape[2:] != x.shape[2:]: 
            dec6_transp = TF.resize(dec6_transp, size=x.shape[2:]) # dims: tile * tile
        dec6_double = self.dec6_double(dec6_transp) # dims: tile * tile

        final_conv =  self.final_conv(dec6_double) # dims: tile * tile

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv


# UNet with ResNet34 encoder
class UNet_Resnet34(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        resnet34_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1, progress=True)   
        modules = list(resnet34_model.children())

        if train_head_only:
            for parameter in resnet34_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in resnet34_model.parameters():
                parameter.requires_grad = True

        self.relu = nn.ReLU(inplace=True)

        self.encoder_blocks = nn.ModuleList(self.get_stages(modules))
        
        self.middle_part = double_conv(512, 1024)
        
        self.dec1_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1_double = double_conv(1024, 512)
        self.dec2_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(512, 256)
        self.dec3_transp = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(256, 128)
        self.dec4_transp = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(128, 64)
        self.dec5_transp = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec5_double = double_conv(128, 64)
        self.dec6_transp = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec6_double = double_conv(32, 32)
        
        self.final_conv = nn.Conv2d(32, self.out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=1024, out_channels=out_channels, pooldim=1)
    
    def get_stages(self, modules):
        return [
            nn.Sequential(*modules[:3]),
            nn.Sequential(*modules[3:5]),
            modules[5],
            modules[6],
            modules[7]
        ]

    def forward(self, x):

        # Encoding part
        x = self.input_adapter(x) # dims: tile * tile

        layer0 = self.encoder_blocks[0](x) # dims: tile/2 * tile/2
        layer1 = self.encoder_blocks[1](layer0) # dims: tile/4 * tile/4
        layer2 = self.encoder_blocks[2](layer1) # dims: tile/8 * tile/8
        layer3 = self.encoder_blocks[3](layer2) # dims: tile/16 * tile/16
        layer4 = self.encoder_blocks[4](layer3) # dims: tile/32 * tile/32

        # Middle part
        middle_part = self.middle_part(layer4) # dims: tile/32 * tile/32

        # Decoding part
        dec1_transp = self.dec1_transp(middle_part) # dims: tile/16 * tile/16
        if dec1_transp.shape[2:] != layer4.shape[2:]:
            dec1_transp = TF.resize(dec1_transp, size=layer4.shape[2:])  # dims: tile/32 * tile/32
        dec1_cat = torch.cat([layer4, dec1_transp], 1) # dims: tile/32 * tile/32
        dec1_double = self.dec1_double(dec1_cat) # dims: tile/32 * tile/32

        dec2_transp = self.dec2_transp(dec1_double) # dims: tile/16 * tile/16
        if dec2_transp.shape[2:] != layer3.shape[2:]: # dims: tile/16 * tile/16
            dec2_transp = TF.resize(dec2_transp, size=layer3.shape[2:])
        
        dec2_cat = torch.cat([layer3, dec2_transp], 1) # dims: tile/16 * tile/16
        dec2_double = self.dec2_double(dec2_cat) # dims: tile/16 * tile/16

        dec3_transp = self.dec3_transp(dec2_double) # dims: tile/8 * tile/8
        if dec3_transp.shape[2:] != layer2.shape[2:]:
            dec3_transp = TF.resize(dec3_transp, size=layer2.shape[2:])
        dec3_cat = torch.cat([layer2, dec3_transp], 1) # dims: tile/8 * tile/8
        dec3_double = self.dec3_double(dec3_cat) # dims: tile/8 * tile/8

        dec4_transp = self.dec4_transp(dec3_double) # dims: tile/4 * tile/4
        if dec4_transp.shape[2:] != layer1.shape[2:]: # dims: tile/4 * tile/4
            dec4_transp = TF.resize(dec4_transp, size=layer1.shape[2:])
        dec4_cat = torch.cat([layer1, dec4_transp], 1) # dims: tile/4 * tile/4
        dec4_double = self.dec4_double(dec4_cat) # dims: tile/4 * tile/4

        dec5_transp = self.dec5_transp(dec4_double) # dims: tile/2 * tile/2
        if dec5_transp.shape[2:] != layer0.shape[2:]: # dims: tile/2 * tile/2
            dec5_transp = TF.resize(dec5_transp, size=layer0.shape[2:])
        dec5_cat = torch.cat([layer0, dec5_transp], 1) # dims: tile/2 * tile/2
        dec5_double = self.dec5_double(dec5_cat) # dims: tile/2 * tile/2

        dec6_transp = self.dec6_transp(dec5_double) # dims: tile * tile
        if dec6_transp.shape[2:] != x.shape[2:]: 
            dec6_transp = TF.resize(dec6_transp, size=x.shape[2:]) # dims: tile * tile
        dec6_double = self.dec6_double(dec6_transp) # dims: tile * tile

        final_conv =  self.final_conv(dec6_double) # dims: tile * tile

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv


# UNet with ResNet50 encoder
class UNet_Resnet_50(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()
        
        resnet50_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, progress=True)   
        modules = list(resnet50_model.children())

        if train_head_only:
            for parameter in resnet50_model.parameters():
                parameter.requires_grad = False
        else:
            for parameter in resnet50_model.parameters():
                parameter.requires_grad = True

        self.relu = nn.ReLU(inplace=True)

        self.encoder_blocks = nn.ModuleList(self.get_stages(modules))
        
        self.middle_part = double_conv(2048, 4096)
    
        self.dec1_transp = nn.ConvTranspose2d(4096, 2048, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.dec1_double = double_conv(4096, 2048)
        self.dec2_transp = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2_double = double_conv(2048, 1024)
        self.dec3_transp = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3_double = double_conv(1024, 512)
        self.dec4_transp = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec4_double = double_conv(512, 256)
        self.dec5_transp = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec5_double = double_conv(128, 64)
        self.dec6_transp = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec6_double = double_conv(32, 32)
        
        self.final_conv = nn.Conv2d(32, self.out_channels, kernel_size=1)

        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(4096, out_channels, pooldim=1)

    def get_stages(self, modules):
        return [
            nn.Sequential(*modules[:3]),
            nn.Sequential(*modules[3:5]),
            modules[5],
            modules[6],
            modules[7]
        ]
        
    def forward(self, x):
    
        # Encoding part
        x = self.input_adapter(x) # dims: tile * tile

        layer0 = self.encoder_blocks[0](x) # dims: tile/2 * tile/2
        layer1 = self.encoder_blocks[1](layer0) # dims: tile/4 * tile/4
        layer2 = self.encoder_blocks[2](layer1) # dims: tile/8 * tile/8
        layer3 = self.encoder_blocks[3](layer2) # dims: tile/16 * tile/16
        layer4 = self.encoder_blocks[4](layer3) # dims: tile/32 * tile/32

        # Middle part
        middle_part = self.middle_part(layer4) # dims: tile/32 * tile/32

        # Decoding part
        dec1_transp = self.dec1_transp(middle_part) # dims: tile/32 * tile/32
        if dec1_transp.shape[2:] != layer4.shape[2:]:
            dec1_transp = TF.resize(dec1_transp, size=layer4.shape[2:])  # dims: tile/32 * tile/32
        dec1_cat = torch.cat([layer4, dec1_transp], 1) # dims: tile/32 * tile/32
        dec1_double = self.dec1_double(dec1_cat) # dims: tile/32 * tile/32

        dec2_transp = self.dec2_transp(dec1_double) # dims: tile/16 * tile/16
        if dec2_transp.shape[2:] != layer3.shape[2:]: # dims: tile/16 * tile/16
            dec2_transp = TF.resize(dec2_transp, size=layer3.shape[2:])
        dec2_cat = torch.cat([layer3, dec2_transp], 1) # dims: tile/16 * tile/16
        dec2_double = self.dec2_double(dec2_cat) # dims: tile/16 * tile/16

        dec3_transp = self.dec3_transp(dec2_double) # dims: tile/8 * tile/8
        if dec3_transp.shape[2:] != layer2.shape[2:]:
            dec3_transp = TF.resize(dec3_transp, size=layer2.shape[2:]) # dims: tile/8 * tile/8
        dec3_cat = torch.cat([layer2, dec3_transp], 1) # dims: tile/8 * tile/8
        dec3_double = self.dec3_double(dec3_cat) # dims: tile/8 * tile/8

        dec4_transp = self.dec4_transp(dec3_double) # dims: tile/4 * tile/4
        if dec4_transp.shape[2:] != layer1.shape[2:]: # dims: tile/4 * tile/4
            dec4_transp = TF.resize(dec4_transp, size=layer1.shape[2:])
        dec4_cat = torch.cat([layer1, dec4_transp], 1) # dims: tile/4 * tile/4
        dec4_double = self.dec4_double(dec4_cat) # dims: tile/4 * tile/4

        dec5_transp = self.dec5_transp(dec4_double) # dims: tile/2 * tile/2
        if dec5_transp.shape[2:] != layer0.shape[2:]: # dims: tile/2 * tile/2
            dec5_transp = TF.resize(dec5_transp, size=layer0.shape[2:])
        dec5_cat = torch.cat([layer0, dec5_transp], 1) # dims: tile/2 * tile/2
        dec5_double = self.dec5_double(dec5_cat) # dims: tile/2 * tile/2

        dec6_transp = self.dec6_transp(dec5_double) # dims: tile * tile
        if dec6_transp.shape[2:] != x.shape[2:]: 
            dec6_transp = TF.resize(dec6_transp, size=x.shape[2:]) # dims: tile * tile
        dec6_double = self.dec6_double(dec6_transp) # dims: tile * tile
        
        final_conv = self.final_conv(dec6_double)
        
        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(middle_part))
            return final_conv, classifier_logits
        else:
            return final_conv

class DeepLabV3(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        
        super().__init__()
        
        self.name = 'Deeplabv3'
        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != 3):
            self.input_adapter = double_conv(in_channels, 3)
        else:
            self.input_adapter = nn.Identity()

        self.deeplabv3 = models.segmentation.deeplabv3_resnet50(weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1, progress=True)
        # self.deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True) 
        self.deeplabv3.aux_classifier = None
        self.deeplabv3.classifier =  models.segmentation.deeplabv3.DeepLabHead(in_channels=2048, num_classes=out_channels)
        # freeze encoder weights 
        self._set_trainable(train_head_only)
        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.deeplabv3.aux_classifier =  nn.Identity()
            self.binary_classifier = FCNHead(1024, out_channels, pooldim=1)
        else:
            self.deeplabv3.aux_classifier = None

    def _set_trainable(self, train_head_only):
        self.train_head_only = train_head_only
        if train_head_only:
            for name, parameter in self.deeplabv3.named_parameters():
                if 'backbone' in name:
                    parameter.requires_grad_(False)
                    parameter.grad = None
        else:
            for name, parameter in self.deeplabv3.named_parameters():
                if 'backbone' in name:  
                    parameter.requires_grad_(True)
                    parameter.grad = torch.zeros_like(parameter)

    def forward(self, x):
        if self.domain_classifier:
            res = self.deeplabv3(x)
            output, features = res['out'], res['aux']
            classifier_logits = self.binary_classifier(reverse_gradients(features))
            return output, classifier_logits
        else:
            output = self.deeplabv3(x)['out']
            return output
            
# class SwinFormer(nn.Module):
#     '''
#     SwinFormer model - this is a segmentation model based on the Swin Transformer architecture. 
#     It is a transformer-based architecture that uses an encoder-decoder architecture. 
#     More specifically, we use the Swin Transformer model from the Hugging Face model hub, which is pre-trained on the ImageNet dataset, 
#     as an encoder, and then we apply a custom CNN decoder.

#     args:

#     in_channels: int, number of input channels
#     out_channels: int, number of output channels
#     train_head_only: bool, if True, only the head of the model - the CNN decoder - is trained
#     domain_classifier: bool, if True, a domain classifier is added to the model
#     '''
#     def __init__(self, in_channels=3, out_channels=1, train_head_only=False, domain_classifier=False):
#         super(SwinFormer, self).__init__()
        
#         self.name = 'SwinFormer'
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         if self.in_channels != 3:
#             self.input_adapter = nn.Sequential(
#                 nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, kernel_size=3, padding=1),
#                 nn.ReLU()
#             )
#         else:
#             self.input_adapter = nn.Identity()

#         self.encoder = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
#         self.decoder = self.Decoder()

#         if train_head_only:
#             for parameter in self.encoder.parameters():
#                 parameter.requires_grad = False
        
#         self.domain_classifier = domain_classifier
#         if self.domain_classifier:
#             self.binary_classifier = FCNHead(in_channels=24, out_channels=self.out_channels, dim_reduction=4)


#     def forward(self, x):
#         x = self.input_adapter(x)
#         x = self.encoder(x).last_hidden_state

#         # Reshape the output of the encoder to fit the decoder
#         batch_size, seq_len, _ = x.shape
#         h = w = int(sqrt(seq_len))
#         x = x.view(batch_size, 24, h*8, w*8)
#         if self.domain_classifier:
#             classifier_logits = self.binary_classifier(reverse_gradients(x))
#             return self.decoder(x), classifier_logits
        
#         return self.decoder(x)

#     class Decoder(nn.Module):
#         '''
#         Decoder class - this is a custom CNN decoder that is used in the SwinFormer model.
#         '''

#         def __init__(self):
#             super(SwinFormer.Decoder, self).__init__()

#             self.conv1 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
#             self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#             self.conv2 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
#             self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#             self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#             self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#             self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

#         def forward(self, x):
#             h = int(x.shape[2]//8)
#             w = int(x.shape[3]//8)

#             x = F.relu(self.conv1(x))
#             x = self.up1(x)
#             x = F.relu(self.conv2(x))
#             x = self.up2(x)
#             x = F.relu(self.conv3(x))
#             x = self.up3(x)
#             x = self.final_conv(x)

#             x = F.adaptive_avg_pool2d(x, (h*32, w*32))
#             return x
        

class SwinFormer(nn.Module):
    '''
    SwinFormer segmentation model using Swinv2 encoder and a lightweight CNN decoder.
    Args:
        in_channels (int): Number of input image channels (default 3).
        out_channels (int): Number of segmentation classes (default 1 for binary).
        train_head_only (bool): If True, freeze encoder and train only decoder.
        domain_classifier (bool): If True, add a domain classification head.
        pretrained_model_name (str): HuggingFace model name for Swinv2.
    '''
    def __init__(self, in_channels=3, out_channels=1, train_head_only=False,
                 domain_classifier=False, pretrained_model_name="microsoft/swinv2-large-patch4-window12-192-22k"):
        super(SwinFormer, self).__init__()
        self.name = 'SwinFormer'
        self.out_channels = out_channels
        self.domain_classifier = domain_classifier

        # Adapt input channels if not 3
        if in_channels != 3:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(3, 3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.input_adapter = nn.Identity()

        # Load Swinv2 encoder
        self.encoder = Swinv2Model.from_pretrained(pretrained_model_name)
        self.hidden_dim = self.encoder.config.hidden_size   # e.g., 1536 for large

        if train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder: simple upsampling path
        # Input feature map channels = hidden_dim
        self.decoder = nn.Sequential(
            # Reduce channels for efficiency
            nn.Conv2d(self.hidden_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, out_channels, kernel_size=1)
        )

        if self.domain_classifier:
            # use the deepest feature map (after encoder) for domain classification
            self.binary_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, out_channels)
            )

    def forward(self, x, return_features=False):
        """
        Args:
            x: input tensor of shape (B, C, H, W)
            return_features: if True, also return intermediate features (for domain classifier)
        Returns:
            segmentation logits of shape (B, out_channels, H, W)
            (optionally) domain classifier logits if domain_classifier is True and return_features is False
        """
        B, _, orig_h, orig_w = x.shape
        x = self.input_adapter(x)

        outputs = self.encoder(x)
        last_hidden = outputs.last_hidden_state   # (B, num_patches, hidden_dim)

        num_patches = last_hidden.shape[1]
        # compute grid size (here we assuming square patches)
        h = w = int(sqrt(num_patches))
        # some inputs may not produce exact square; fallback to patch arrangement
        if h * w != num_patches:
            patch_size = self.encoder.config.patch_size
            h = orig_h // patch_size
            w = orig_w // patch_size
            if h * w != num_patches:
                raise ValueError(f"Patch grid mismatch: {num_patches} vs {h*w}")
        feature_map = last_hidden.permute(0, 2, 1).view(B, self.hidden_dim, h, w)

        # domain classifier branch (if needed and called separately)
        if self.domain_classifier and return_features:
            return feature_map

        # decoder produces logits at original resolution
        seg_logits = self.decoder(feature_map)
        # ensure exact size (in case of rounding)
        if seg_logits.shape[-2:] != (orig_h, orig_w):
            seg_logits = F.interpolate(seg_logits, size=(orig_h, orig_w),
                                       mode='bilinear', align_corners=False)

        if self.domain_classifier:
            # global pooling on encoder features for domain classification
            domain_logits = self.binary_classifier(feature_map)
            return seg_logits, domain_logits
        else:
            return seg_logits


class Segformer(nn.Module):
    '''
    Segformer model - this is a segmentation model based on the Segformer architecture. 
    It is a transformer-based architecture that uses a convolutional stem and a transformer encoder-decoder architecture. 
    We use the pre-trained model from the NVIDIA NGC model hub, which is pre-trained on the ADE20K dataset.
  
    args:
  
    in_channels: int, number of input channels
    out_channels: int, number of output channels
    preprocess: bool, if True, the input image is preprocessed using the SegformerImageProcessor
    domain_classifier: bool, if True, a domain classifier is added to the model
    train_head_only: bool, if True, only the head of the model is trained
    '''

    def __init__(self, in_channels=3, out_channels=1, preprocess=True, domain_classifier=False, train_head_only=False):
        super(Segformer, self).__init__()
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        config.strides = [1, 2, 2, 2] # Due to output channes mismatch (we use 1 (binary classification) instead of 150 (ADE20K classes) in which the model was trained)
        config.num_labels = out_channels
        config.output_hidden_states = True # Used in the domain adaptation task

        # Model Definition
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True, config=config)

        # Train Head Only
        if train_head_only:
            #Freeze all the model paremeters
            for param in self.model.parameters():
                param.requires_grad = False
            #Unfreeze the segmentation head
            for param in self.model.segmentation_head.parameters():
                param.requires_grad = True

        # Preprocessing
        self.preprocess = preprocess
        if self.preprocess:
            self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
        # Domain Adaptation
        self.domain_classifier = domain_classifier
        if self.domain_classifier:
                self.binary_classifier = FCNHead(in_channels=256, out_channels=out_channels, pooldim=1)

    def forward(self, x):
        dim = x.shape[3]
        
        if self.preprocess:
        
            # Normalize the input image
            if x.min() < 0 or x.max() > 1:
                x = (x - x.min()) / (x.max() - x.min())
            
            # Preprocess the image 
            # NOTES: We rescaled the image in the previous line, so we se do_rescale=False to avoid rescaling the image again
            #        We also set do_resize=False to because normaly the preprocessor it resizes the image to 512x512, which is too large for the ram of the GPU
            #        If this image could fit, it would be ideal because the model was trained with this size. Since it can't for the time being due to hardware limitations,
            #        we will have to resize the image to 256x256 and smaller. Nvidia does not provide a model trained with this size unfortunately, since 512x512 is the smallest size they provide,
            #        but it should work fine with lower resolutions as well.

            _input_device = x.device  # capture device before processor moves tensor to CPU
            x = self.processor(images=x, return_tensors='pt', do_resize=False, do_rescale=False).pixel_values
            if torch.cuda.is_available():
                    x = x.to(_input_device)  # move back to input device (avoids StopIteration from empty params in DP replica)

        outputs = self.model(x)
        logits = outputs.logits

        if self.preprocess:
            logits = F.adaptive_avg_pool2d(logits, (dim, dim)) # In case we resized the image, we need to resize the output to the original size

        if self.domain_classifier:
            features = outputs.hidden_states[-1]
            classifier_logits = self.binary_classifier(reverse_gradients(features))
            return logits, classifier_logits
        
        else:
            return logits

##NEW DEVELOPMENT, integrated Unet V2 (with transformer)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # [(h+2p - d(k-1))/s ]+1 = [h+2p -6]

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #(b,c,h,w)# taking mean, max in channel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):# for channel 
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.backbone = pvt_v2_b2()

        if pretrain_path is None:
            warnings.warn('please provide the pretrained pvt model. Not using pretrained model.')
        elif not os.path.isfile(pretrain_path):
            warnings.warn(f'path: {pretrain_path} does not exists. Not using pretrained model.')
        else:
            print(f"using pretrained file: {pretrain_path}")
            save_model = torch.load(pretrain_path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)

            self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)  # (x: 3, 352, 352)
        return f1, f2, f3, f4


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):#xs:tensor
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class UNetV2(nn.Module):
    """
    use SpatialAtt + ChannelAtt
    """
    def __init__(self, in_channels=3, out_channels=1,train_head_only=False,domain_classifier=False, channel = 32, deep_supervision=True, pretrained_path=None):
        super().__init__()
        self.train_head_only = train_head_only
        self.deep_supervision = deep_supervision
        self.domain_classifier = domain_classifier

        if pretrained_path is None:
            pretrained_path = os.path.join(os.path.dirname(__file__), "pvt_v2_b2.pth")
        self.encoder = Encoder(pretrained_path)

        if self.train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad=False
        
        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(64, channel, 1)#in_planes, out_planes, kernel_size
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)

        self.sdi_1 = SDI(channel)
        self.sdi_2 = SDI(channel)
        self.sdi_3 = SDI(channel)
        self.sdi_4 = SDI(channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, out_channels, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        if self.domain_classifier:
            # The number of input channels for FCNHead should be the same as the number of feature map channels we use for classification.
            # Here we choose to attach the classification header to the deepest SDI module output f41, which has exactly the number of channels' channel '.
            self.binary_classifier = FCNHead(in_channels=channel, out_channels=out_channels, pooldim=1)

    def forward(self, x):
        seg_outs = []
        f1, f2, f3, f4 = self.encoder(x)

        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        segmentation_output = seg_outs[-1]

        # deep supervision has been currently deprecated due to training infra not match
        # if self.deep_supervision:
        #     segmentation_output = seg_outs[::-1]
        # else:
        #     segmentation_output = seg_outs[-1]

        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(f41))
            return segmentation_output, classifier_logits
        else:
            return segmentation_output

# new development, design a code interface for DINOv3
# only supports 1024 currently, a specified DINOV3 model trained on satellite images
class DINOv3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True, model_config: dict = None, repo_path: str = None, cache_dir: str = None):
        # Expected a model_config: dict to be passed (needs key 'url')
        super().__init__()
        if model_config is None or repo_path is None:
            raise ValueError("DINOv3 model requires 'model_config' and 'repo_path' to be provided.")
        # using dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
        model_url = model_config['url']
        hub_model_name = model_config['hub_model_name']
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/hub/checkpoints')
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, file_name)
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            response = requests.head(model_url, allow_redirects=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            #download chunk size
            block_size = 1024 # 1KB
            
            with requests.get(model_url, stream=True) as r, open(model_path, "wb") as f, tqdm(
                unit="B", 
                unit_scale=True, 
                unit_divisor=1024, 
                total=total_size_in_bytes, 
                desc=file_name 
            ) as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete.")
        # need to specify your own git clone repo here
        self.encoder = torch.hub.load(repo_path, hub_model_name,source='local', weights=model_path)# does not need to code adapter here since this feature is 1024 size
        self.name = 'unet_dinov3'
        self.out_channels = out_channels
        # output dim equals 1024
        self.dinov3_feature_dim = model_config['feature_dim']
        if train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dinov3_feature_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            # use feature dim as domain_classifier, dinov3_feature_dim
            self.binary_classifier = FCNHead(in_channels=self.dinov3_feature_dim, out_channels=out_channels, pooldim=1)
    
    def forward(self,x):
        patch_tokens = self.encoder.get_intermediate_layers(x, n=1)[0]# register_hooks=False, since we dont need features from skip connections
        # returns cls token and patch token
        #ViT arch

        _, num_patches, _ = patch_tokens.shape
        h = w = int(num_patches ** 0.5)
        
        if h * w != num_patches:
            patch_size = 16  # patch size for dinov3_vitl16 is 16
            input_h, input_w = x.shape[2:]
            h = input_h // patch_size
            w = input_w // patch_size

        #check again if num_patches equals reconstructed img patch size
        if h * w != num_patches:
            raise ValueError(f"Number of patches ({num_patches}) does not match expected size for input {input_h}x{input_w} with patch size {patch_size}. HxW: {h}x{w}")

        patch_tokens = patch_tokens.permute(0, 2, 1) # (B, 1024, num_patches)
        feature_map = patch_tokens.view(-1, self.dinov3_feature_dim, h, w) # (B, 1024, H/16, W/16)
        segmentation_output = self.decoder(feature_map)
        segmentation_output = F.interpolate(segmentation_output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.domain_classifier:
            cls_token_for_head = cls_token.unsqueeze(-1).unsqueeze(-1)# (B,1024,1,1)
            classifier_logits = self.binary_classifier(reverse_gradients(cls_token_for_head))
            return segmentation_output, classifier_logits
        else:
            return segmentation_output

# A more generalized model trained on a comprehensive image datasets
class DINOv3_Small(nn.Module):
    """
    A variant of UNet_DINOv3 with adapter layer
    Embedding features output: 384
    """
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()
        # using dinov3_vits16_pretrain_lvd1689m-08c60483.pth
        model_url = 'https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3pjdWNqNTBna2I5dHdoejQ0YXRqbWczIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTYxNTk2NTZ9fX1dfQ__&Signature=a-sq1JjffDScyPO52iLzI3qxbADZO507eHDb6TaO0bTKJcfZ3-tPyqPsV0K47PfzBCs8og0uu-0jUSV0eYQxxPqlfIFjDH3GDpX25iuwTT%7ESef0b1fH6CyY-sYabFfJqUa6hxAeTNzZ07blBPxFf2nrPqUKMUiC-972Pg2KtGZ1XsPD-lU89c1JmBBnAsG%7E26M56-OqOhzgUTJbFbuTHT4ayBugL6iyrvGp18u1Gx%7Ent4KYyyotq8zMxBlb99IQMEJBMIwVxsFo74CT8o8hJPqUnBB6sZMt4xcM1VUPtCt5clvRTxy6mGg-JdUiSLis2qTxA4W4LfuKsP9YcYwqgdQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1138153021522372'
        file_name = model_url.split('/')[-1].split('?')[0]
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/hub/checkpoints')
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, file_name)
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            response = requests.head(model_url, allow_redirects=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            #download chunk size
            block_size = 1024 # 1KB
            
            with requests.get(model_url, stream=True) as r, open(model_path, "wb") as f, tqdm(
                unit="B", 
                unit_scale=True, 
                unit_divisor=1024, 
                total=total_size_in_bytes, 
                desc=file_name 
            ) as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete.")
        # need to specify your own git clone repo here
        dinov3_repo_path = '/home/yuting/dinov3'

        self.encoder = torch.hub.load(dinov3_repo_path, 'dinov3_vits16', source='local', weights=model_path)
        self.name = 'unet_dinov3_small'
        self.out_channels = out_channels
        
        self.dinov3_feature_dim = 384
        
        if train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # adapter layer, 384->512
        self.adapter = nn.Conv2d(self.dinov3_feature_dim, 512, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=self.dinov3_feature_dim, out_channels=out_channels, pooldim=1)
    
    def forward(self, x):
        patch_tokens = self.encoder.get_intermediate_layers(x, n=1)[0]
        _, num_patches, _ = patch_tokens.shape
        h = w = int(num_patches ** 0.5)
        
        if h * w != num_patches:
            patch_size = 16  # patch size for dinov3_vitl16 is 16
            input_h, input_w = x.shape[2:]
            h = input_h // patch_size
            w = input_w // patch_size

        #check again if num_patches equals reconstructed img patch size
        if h * w != num_patches:
            raise ValueError(f"Number of patches ({num_patches}) does not match expected size for input {input_h}x{input_w} with patch size {patch_size}. HxW: {h}x{w}")

        patch_tokens = patch_tokens.permute(0, 2, 1)
        feature_map = patch_tokens.view(-1, self.dinov3_feature_dim, h, w)
        adapted_feature_map = self.adapter(feature_map)
        
        segmentation_output = self.decoder(adapted_feature_map)
        segmentation_output = F.interpolate(segmentation_output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.domain_classifier:
            cls_token_for_head = cls_token.unsqueeze(-1).unsqueeze(-1)
            classifier_logits = self.binary_classifier(reverse_gradients(cls_token_for_head))
            return segmentation_output, classifier_logits
        else:
            return segmentation_output

# A more generalized model trained on a comprehensive image datasets
class DINOv3_Base(nn.Module):
    """
    A variant of UNet_DINOv3 with adapter layer
    Embedding features output: 768
    """
    def __init__(self, in_channels=3, out_channels=1, train_head_only=True, domain_classifier=True):
        super().__init__()
        # using dinov3_vits16_pretrain_lvd1689m-08c60483.pth
        model_url = 'https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3pjdWNqNTBna2I5dHdoejQ0YXRqbWczIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTYxNTk2NTZ9fX1dfQ__&Signature=a-sq1JjffDScyPO52iLzI3qxbADZO507eHDb6TaO0bTKJcfZ3-tPyqPsV0K47PfzBCs8og0uu-0jUSV0eYQxxPqlfIFjDH3GDpX25iuwTT%7ESef0b1fH6CyY-sYabFfJqUa6hxAeTNzZ07blBPxFf2nrPqUKMUiC-972Pg2KtGZ1XsPD-lU89c1JmBBnAsG%7E26M56-OqOhzgUTJbFbuTHT4ayBugL6iyrvGp18u1Gx%7Ent4KYyyotq8zMxBlb99IQMEJBMIwVxsFo74CT8o8hJPqUnBB6sZMt4xcM1VUPtCt5clvRTxy6mGg-JdUiSLis2qTxA4W4LfuKsP9YcYwqgdQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1138153021522372'
        file_name = model_url.split('/')[-1].split('?')[0]
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/hub/checkpoints')
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, file_name)
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            response = requests.head(model_url, allow_redirects=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            #download chunk size
            block_size = 1024 # 1KB
            
            with requests.get(model_url, stream=True) as r, open(model_path, "wb") as f, tqdm(
                unit="B", 
                unit_scale=True, 
                unit_divisor=1024, 
                total=total_size_in_bytes, 
                desc=file_name 
            ) as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete.")
        dinov3_repo_path = '/home/yuting/dinov3'
        self.encoder = torch.hub.load(dinov3_repo_path, 'dinov3_vitb16',source='local', weights=model_path)
        self.name = 'unet_dinov3_base'
        self.out_channels = out_channels
        
        self.dinov3_feature_dim = 768
        
        if train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.adapter = nn.Conv2d(self.dinov3_feature_dim, 512, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        self.domain_classifier = domain_classifier
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=self.dinov3_feature_dim, out_channels=out_channels, pooldim=1)
    
    def forward(self, x):
        patch_tokens = self.encoder.get_intermediate_layers(x, n=1)[0]
        _, num_patches, _ = patch_tokens.shape
        h = w = int(num_patches ** 0.5)
        
        if h * w != num_patches:
            patch_size = 16  # patch size for dinov3_vitl16 is 16
            input_h, input_w = x.shape[2:]
            h = input_h // patch_size
            w = input_w // patch_size

        #check again if num_patches equals reconstructed img patch size
        if h * w != num_patches:
            raise ValueError(f"Number of patches ({num_patches}) does not match expected size for input {input_h}x{input_w} with patch size {patch_size}. HxW: {h}x{w}")

        patch_tokens = patch_tokens.permute(0, 2, 1)
        feature_map = patch_tokens.view(-1, self.dinov3_feature_dim, h, w)
        
        adapted_feature_map = self.adapter(feature_map)
        
        segmentation_output = self.decoder(adapted_feature_map)
        segmentation_output = F.interpolate(segmentation_output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.domain_classifier:
            cls_token_for_head = cls_token.unsqueeze(-1).unsqueeze(-1)
            classifier_logits = self.binary_classifier(reverse_gradients(cls_token_for_head))
            return segmentation_output, classifier_logits
        else:
            return segmentation_output


class UNetV2_DINOV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, train_head_only=False, channel=32,
                 deep_supervision=False, pretrained_path=None, dino_feature_dim=1024, domain_classifier = False):
        super().__init__()
        self.train_head_only = train_head_only
        self.deep_supervision = deep_supervision
        self.dino_feature_dim = dino_feature_dim
        self.domain_classifier = domain_classifier

        if pretrained_path is None:
            pretrained_path = os.path.join(os.path.dirname(__file__), "pvt_v2_b2.pth")
        self.encoder = Encoder(pretrained_path)# output should be 512

        if self.train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        pvt_f4_channels = 512
        fusion_in_channels = pvt_f4_channels + self.dino_feature_dim
        self.bottleneck_fusion_block = BasicConv2d(fusion_in_channels, pvt_f4_channels, kernel_size=1)
        
        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()
        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()
        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()
        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()
        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)
        self.sdi_1 = SDI(channel)
        self.sdi_2 = SDI(channel)
        self.sdi_3 = SDI(channel)
        self.sdi_4 = SDI(channel)
        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, out_channels, 1, 1) for _ in range(4)])
        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=channel, out_channels=out_channels, pooldim=1)

    def forward(self, image, dino_features):
        f1, f2, f3, f4 = self.encoder(image)
        
        dino_features_aligned = F.interpolate(dino_features, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        f4_concatenated = torch.cat([f4, dino_features_aligned], dim=1)
        f4_fused = self.bottleneck_fusion_block(f4_concatenated)
        seg_outs = []
        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)
        f4 = self.ca_4(f4_fused) * f4_fused
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)
        
        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))
        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))
        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))
        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))
        
        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear', align_corners=False)

        segmentation_output = seg_outs[-1]
        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(f41))
            return segmentation_output, classifier_logits
        else:
            return segmentation_output


class UNetV2_DINOV3_att(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, train_head_only=False, channel=32,
                 deep_supervision=False, pretrained_path=None, dino_feature_dim=1024, domain_classifier=False):
        super().__init__()
        self.train_head_only = train_head_only
        self.deep_supervision = deep_supervision
        self.dino_feature_dim = dino_feature_dim
        self.domain_classifier = domain_classifier

        if pretrained_path is None:
            pretrained_path = os.path.join(os.path.dirname(__file__), "pvt_v2_b2.pth")
        self.encoder = Encoder(pretrained_path)

        if self.train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # DinoV3 channel attention
        self.dino_channel_attention = DinoChannelAttention(
            dino_feature_dim=dino_feature_dim,
            pvt_channels=[64, 128, 320, 512],
            output_channels=channel
        )
        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()
        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()
        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()
        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()
        
        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)
        
        self.sdi_1 = SDI(channel)
        self.sdi_2 = SDI(channel)
        self.sdi_3 = SDI(channel)
        self.sdi_4 = SDI(channel)
        
        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, out_channels, 1, 1) for _ in range(4)])
        
        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        
        if self.domain_classifier:
            self.binary_classifier = FCNHead(in_channels=channel, out_channels=out_channels, pooldim=1)

    def forward(self, image, dino_features):
        f1, f2, f3, f4 = self.encoder(image)
        
        f1_att, f2_att, f3_att, f4_att = self.dino_channel_attention(dino_features, [f1, f2, f3, f4])
        
        seg_outs = []
        
        f1 = self.ca_1(f1) * f1 * f1_att
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2 * f2_att
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3 * f3_att
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4 * f4_att
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)
        
        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))
        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))
        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))
        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))
        
        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear', align_corners=False)

        segmentation_output = seg_outs[-1]
        if self.domain_classifier:
            classifier_logits = self.binary_classifier(reverse_gradients(f41))
            return segmentation_output, classifier_logits
        else:
            return segmentation_output


class DinoChannelAttention(nn.Module):
    """DinoV3 att block"""
    def __init__(self, dino_feature_dim=1024, pvt_channels=[64, 128, 320, 512], output_channels=32):
        super().__init__()
        
        self.global_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dino_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        # create attention generator for each layer
        self.attention_generators = nn.ModuleList()
        for pvt_channel in pvt_channels:
            self.attention_generators.append(
                nn.Sequential(
                    nn.Linear(256, pvt_channel),
                    nn.Sigmoid()  
                )
            )
        
        self.feature_projectors = nn.ModuleList()
        for pvt_channel in pvt_channels:
            self.feature_projectors.append(
                nn.Sequential(
                    nn.Linear(256, pvt_channel),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, dino_features, pvt_features_list):
        """
        Args:
            dino_features: [B, dino_feature_dim, H, W] - DinoV3 features
            pvt_features_list: List of PVT featurs [f1, f2, f3, f4]
        Returns:
            features list aggrevgated by attention blocks
        """
        batch_size = dino_features.shape[0]
        global_dino = self.global_feature_extractor(dino_features)  # [B, 256]
        
        attention_maps = []
        for i, (pvt_feat, att_gen, proj) in enumerate(zip(
            pvt_features_list, self.attention_generators, self.feature_projectors)):
            
            channel_weights = att_gen(global_dino)  # [B, pvt_channel]
            channel_weights = channel_weights.view(batch_size, -1, 1, 1)  # [B, pvt_channel, 1, 1]

            feature_enhancement = proj(global_dino)  # [B, pvt_channel]
            feature_enhancement = feature_enhancement.view(batch_size, -1, 1, 1)  # [B, pvt_channel, 1, 1]
            
            attended_feature = pvt_feat * channel_weights + feature_enhancement
            attention_maps.append(attended_feature)
        
        return attention_maps


MODELS_REGISTRY = { 
                'unetv2_dinov3_att':UNetV2_DINOV3_att,
                'unet_v2_dinov3': UNetV2_DINOV3,
                'dinov3_768':DINOv3_Base,
                'dinov3_384':DINOv3_Small,
                'dinov3_1024':DINOv3,
                'unet_v2': UNetV2,
                'unet': UNet,
                'unet_vgg11': UNet_VGG11,
                'unet_vgg11_bn': UNet_VGG11_BN,
                'unet_vgg11_bn2': UNet_VGG11_BN2,
                'unet_vgg13_bn': UNet_VGG13_BN,
                'unet_vgg16_bn': UNet_VGG16_BN,
                'unet_vgg19_bn': UNet_VGG19_BN,
                'unet_resnet18': UNet_Resnet18,
                'unet_resnet34': UNet_Resnet34,
                'unet_resnet50': UNet_Resnet_50,
                'deeplabv3':DeepLabV3,
                'swinformer': SwinFormer,
                'segformer': Segformer
}
