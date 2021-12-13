import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.ResNet import resnet18 as resnet
from architectures.SPADE import SPADEResnetBlock, AttentionTransferLayer, BaseNetwork


def printTensorList(data):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
        print(']')


class EdgeGuidedNetwork(BaseNetwork):
    def __init__(self, seg_classes=30):
        super(EdgeGuidedNetwork, self).__init__()
        nf = 64

        self.E = resnet(seg_classes)

        self.attention = AttentionTransferLayer()

        self.G_e_1 = SPADEResnetBlock(nf, nf // 2, seg_classes)
        self.G_e_2 = SPADEResnetBlock(nf // 2, nf // 4, seg_classes)
        self.G_e_3 = SPADEResnetBlock(nf // 4, nf // 8, seg_classes)

        self.G_e_img = nn.Conv2d(nf // 8, 3, 3, padding=1)

        self.G_i_1 = SPADEResnetBlock(nf, nf // 2, seg_classes)
        self.G_i_2 = SPADEResnetBlock(nf // 2, nf // 4, seg_classes)
        self.G_i_3 = SPADEResnetBlock(nf // 4, nf // 8, seg_classes)

        self.G_i_img = nn.Conv2d(nf // 8, 3, 3, padding=1)

        # Semantic Preserving Module
        self.F_c_conv = nn.Conv2d((nf + seg_classes + 3 + 3), seg_classes, 3,
                                  padding=1)
        self.adapt_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sig = nn.Sigmoid()
        self.F_c_d_conv = nn.Conv2d(seg_classes, (nf + seg_classes + 3 + 3), 3,
                                    padding=1)

        self.I_img = nn.Conv2d((nf + seg_classes + 3 + 3), 3, 3, padding=1)

    def forward(self, seg):
        # Parameter sharing encoder
        F_enc = self.E(seg)

        # Attention Guided Image generation (EdgeGAN I)
        F_e_1 = self.G_e_1(F_enc, seg)
        F_i_1 = self.G_e_1(F_enc, seg)
        F_i_1_attn = self.attention(F_e_1, F_i_1)

        F_e_2 = self.G_e_2(F_e_1, seg)
        F_i_2 = self.G_i_2(F_i_1_attn, seg)
        F_i_2_attn = self.attention(F_e_2, F_i_2)

        F_e_3 = self.G_e_3(F_e_2, seg)
        F_i_3 = self.G_i_3(F_i_2_attn, seg)
        F_i_3_attn = self.attention(F_e_3, F_i_3)

        I_e_d = torch.tanh(self.G_e_img(F.leaky_relu(F_e_3, 2e-1)))
        I_i_d = torch.tanh(self.G_i_img(F.leaky_relu(F_i_3_attn, 2e-1)))
        I_d = self.attention(I_e_d, I_i_d)

        # Semantic Preserving Module
        sem_pre_mod = torch.cat([seg, I_e_d, I_d, F_enc], 1)

        F_c = self.F_c_conv(sem_pre_mod)
        gamma = self.adapt_avg(F_c)
        gamma_d = self.sig(gamma)
        F_c_d = (gamma_d * F_c) + F_c
        F_d = self.F_c_d_conv(F_c_d)
        I_dd = torch.tanh(self.I_img(F_d))

        return {'edge': I_e_d,
                'image_init': I_d,
                'image': I_dd}


if __name__ == '__main__':
    seg_classes = 30
    net = EdgeGuidedNetwork(seg_classes)
    net.init_weights()
    print(net)
    seg = torch.Tensor(4, seg_classes, 512, 256)
    out = net(seg)
    printTensorList(out)

