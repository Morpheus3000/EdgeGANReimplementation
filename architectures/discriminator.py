"""
Code adapted from https://github.com/NVlabs/SPADE
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from architectures.SPADE import BaseNetwork
from synchronised_batch_norm.batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from architectures.SPADE import BaseNetwork

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, in_channels=34+1): 
        super().__init__()
        num_D = 2
        self.in_channels = in_channels
        for i in range(num_D):
            subnetD = self.create_single_discriminator()
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self):
        netD = NLayerDiscriminator(self.in_channels)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        # get_intermediate_features = True
        for name, D in self.named_children():
            out = D(input)
            # if not get_intermediate_features:
            #     out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self, in_channels):
        super().__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64
        n_layers_D = 3 # parser default is 4

        norm_layer = get_nonspade_norm_layer()
        sequence = [[nn.Conv2d(in_channels, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = True
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer():
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        layer = spectral_norm(layer)

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


if __name__ == '__main__':
    # generate
    from Network import *
    seg_classes = 30
    net = EdgeGuidedNetwork(seg_classes)
    net.init_weights()
    # print(net)
    seg = torch.Tensor(4, seg_classes, 512, 256)
    out = net(seg)
    I_e_d, I_d, I_dd = out['edge'], out['image_init'], out['image']
    
    I_e = torch.randn_like(I_e_d)
    I = torch.randn_like(I_d)
	
    # discriminator part
    D_edge = MultiscaleDiscriminator(seg_classes+1) # if edgemap channel is 1
    D_image = MultiscaleDiscriminator(seg_classes+3)
    pred_edge_fake = D_edge(torch.cat([seg, I_e_d], dim=1))
    pred_image_fake1 = D_image(torch.cat([seg, I_d], dim=1))
    pred_image_fake2 = D_image(torch.cat([seg, I_dd], dim=1))
    pred_edge_real = D_edge(torch.cat([seg, I_e], dim=1))
    pred_image_real = D_image(torch.cat([seg, I], dim=1))
    
    # printTensorList(out)
