import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from utils.util import *
import torch.nn.functional as F
import numpy as np
import cv2
import copy
from .amm import GaussProjection, Chi_square_distribution, T_distribution, F_distribution, Sigmoidcustom

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCam(nn.Module):
    def __init__(self, block, layers, args, large_feature_map=True):
        super(ResNetCam, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_cls = make_cls_classifier(2048,200)

        self.classifier_loc = MuitipleRoadLoc(1024,200)
        initialize_weights(self.modules(), init_mode='xavier')
    def forward(self, x, label=None, N=1):
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)
        layer4_copy = copy.deepcopy(self.layer4)

        batch = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x_3 = x.clone()
        
        x = F.max_pool2d(x, kernel_size=2)
        x = self.layer4(x)
        
        x = self.classifier_cls(x)
        self.feature_map = x

        self.score_1 = self.avg_pool(x).squeeze(-1).squeeze(-1)

# p_label 
        if N == 1:
            p_label = label.unsqueeze(-1)
        else:
            _, p_label = self.score_1.topk(N, 1, True, True)

# x_sum   
        self.x_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.x_sum[i] = self.score_1[i][label[i]]
    
## x_saliency    
        x_saliency_all = self.classifier_loc(x_3,p_label)
        x_saliency = torch.zeros(batch, 1, 28, 28).cuda()
        for i in range(batch):
            x_saliency[i][0] = x_saliency_all[i][p_label[i]].mean(0)
        self.x_saliency = x_saliency
##  erase
        x_saliency=torch.max(2*self.x_saliency-1,0*x_saliency)
        x_erase = x_3.detach() *  (x_saliency)
        x_erase = F.max_pool2d(x_erase, kernel_size=2)
        x_erase = layer4_copy(x_erase) 
        x_erase = classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).view(x_erase.size(0), -1)

## x_erase_sum
        self.x_erase_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.x_erase_sum[i] = x_erase[i][label[i]]

## score_2
        x = self.feature_map * nn.AvgPool2d(2)(x_saliency)
        self.score_2 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        return self.score_1, self.score_2 

    def bas_loss(self):
        batch = self.x_sum.size(0)
        x_sum = self.x_sum.clone().detach()
        x_res = self.x_erase_sum
        res = 0.75-x_res / (x_sum + 1e-8)
        res[res<0] = 0 ## or 1

        x_saliency =  self.x_saliency
        x_saliency =  x_saliency.clone().view(batch, -1)
        x_saliency = x_saliency.mean(1)  
        
        loss = res  + x_saliency * 1.2
        loss = loss.mean(0) 
        return loss


    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers
        
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

def load_pretrained_model(model):
    strict_rule = True

    state_dict = torch.load('Model/resnet50-19c8e357.pth')

    state_dict = remove_layer(state_dict, 'fc')
    strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def model(args, pretrained=True):
    model = ResNetCam(Bottleneck, [3, 4, 6, 3], args)
    if pretrained:
        model = load_pretrained_model(model)
    return model
def make_cls_classifier(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, out_planes, kernel_size=1, padding=0),
        nn.ReLU(inplace=True),
    )
def make_loc_classifier(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),  ## num_classes
        nn.Sigmoid(),
    )
class MuitipleRoad(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(MuitipleRoad,self).__init__()
        self.cls_classifier_A=make_cls_classifier(in_plane,out_plane)
        self.cls_classifier_B=make_cls_classifier(in_plane,out_plane)
        self.cls_classifier_C = make_cls_classifier(in_plane, out_plane)
        self.cls_classifier_D = make_cls_classifier(in_plane, out_plane)
    def forward(self,x,label):
        feat_map_a=self.cls_classifier_A(x)
        attention=self.get_attention(feat_map_a,label)
        change_feat_B=self.erase_attention(x,attention,'Gauss')
        feat_map_b=self.cls_classifier_B(change_feat_B)
        change_feat_C=self.erase_attention(x,attention,'Sigmoid')
        feat_map_c=self.cls_classifier_C(change_feat_C)
        change_feat_D=self.erase_attention(x,attention,'pi')
        feat_map_d=self.cls_classifier_D(change_feat_D)
        return self.normalize_tensor(feat_map_a+feat_map_b+feat_map_c+feat_map_d)
    def get_attention(self, feat_map, label, normalize=True):
        """
        :return: return attention size (batch, 1, h, w)
        """
        label = label.long()
        b = feat_map.size(0)
        attention = feat_map.detach().clone().requires_grad_(True)[range(b), label.data, :, :]
        attention = attention.unsqueeze(1)
        if normalize:
            attention = self.normalize_tensor(attention)
        return attention
    def erase_attention(self, feature, attention_map,type):
        if type=='Gauss':
            mean=torch.mean(attention_map)
            std=torch.std(attention_map)
            mask=self.normalize_tensor(GaussProjection(attention_map,mean,std))
        elif type=='Sigmoid':
            mask=self.normalize_tensor(Sigmoidcustom(attention_map))
        elif type=='pi':
            mask = self.normalize_tensor(T_distribution(attention_map, 5))
        erased_feature = feature * mask
        return erased_feature
    def normalize_tensor(self,x):
        map_size = x.size()
        aggregated = x.view(map_size[0], map_size[1], -1)
        minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)
        maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)
        normalized = torch.div(aggregated - minimum, maximum - minimum)
        normalized = normalized.view(map_size)

        return normalized
class MuitipleRoadLoc(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(MuitipleRoadLoc,self).__init__()
        self.loc_classifier_A=make_loc_classifier(in_plane,out_plane)
        self.loc_classifier_B = make_loc_classifier(in_plane, out_plane)
        self.loc_classifier_C = make_loc_classifier(in_plane, out_plane)
        self.loc_classifier_D = make_loc_classifier(in_plane, out_plane)
    def forward(self,x,p_label):
        feat_map_a=self.loc_classifier_A(x)
        attention=self.get_attention(feat_map_a,p_label)
        change_feat_B=self.erase_attention(x,attention,'Gauss')
        feat_map_b=self.loc_classifier_B(change_feat_B)
        change_feat_C=self.erase_attention(x,attention,'Sigmoid')
        feat_map_c=self.loc_classifier_C(change_feat_C)
        change_feat_D=self.erase_attention(x,attention,'pi')
        feat_map_d=self.loc_classifier_D(change_feat_D)
        return self.normalize_tensor(feat_map_a+feat_map_b+feat_map_c+feat_map_d)
    def get_attention(self, feat_map, p_label, normalize=True):
        """
        :return: return attention size (batch, 1, h, w)
        """
        batch = feat_map.size(0)
        attention = torch.zeros(batch, 1, feat_map.size(-2), feat_map.size(-1)).cuda()
        for i in range(batch):
            attention[i][0] = feat_map.detach().clone().requires_grad_(True)[i][p_label[i]].mean(0)
        if normalize:
            attention = self.normalize_tensor(attention)
        return attention
    def erase_attention(self, feature, attention_map,type):
        if type=='Gauss':
            mean=torch.mean(attention_map)
            std=torch.std(attention_map)
            mask=self.normalize_tensor(GaussProjection(attention_map,mean,std))
        elif type=='Sigmoid':
            mask=self.normalize_tensor(Sigmoidcustom(attention_map))
        elif type=='pi':
            mask = self.normalize_tensor(T_distribution(attention_map, 5))
        erased_feature = feature * mask
        return erased_feature
    def normalize_tensor(self,x):
        map_size = x.size()
        aggregated = x.view(map_size[0], map_size[1], -1)
        minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)
        maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)
        normalized = torch.div(aggregated - minimum, maximum - minimum+ 1e-10)
        normalized = normalized.view(map_size)
        return normalized