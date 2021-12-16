import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.VGGNet import VGG19


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def to(self, device=None):
        self.device = device
        super().to(device)
        return self

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label).to(self.device)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label).to(self.device)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0).to(self.device)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def to(self, device=None):
        self.vgg.to(device)
        self.criterion.to(device)
        super().to(device)
        return self

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# GAN feature loss
class FeatLoss(nn.Module):
    def __init__(self):
        self.criterionFeat = torch.nn.L1Loss()
        self.lambda_feat = 10.0

    def to(self, device=None):
        self.device = device
        super().to(device)
        return self

    def forward(self, pred_fake, pred_real):
        GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
        num_D = len(pred_fake)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.lambda_feat / num_D
        return GAN_Feat_loss


class MultiModalityDiscriminatorLoss(nn.Module):
    def __init__(self, edge_discriminator, image_discriminator, seg_classes=34,
                 lambda_feat=10, lambda_vgg=10, lambda_gan=1, lambd=2):
        super(MultiModalityDiscriminatorLoss, self).__init__()
        self.edge_discriminator = edge_discriminator
        self.image_discriminator = image_discriminator
        self.lambd = lambd
        self.crit_GAN = GANLoss('original')
        self.crit_Feat = torch.nn.L1Loss()
        self.crit_Vgg = VGGLoss()
        self.lambda_vgg = lambda_vgg
        self.lambda_feat = lambda_feat
        self.lambda_gan = lambda_gan

    def to(self, device=None):
        self.device = device
        self.edge_discriminator.to(device)
        self.image_discriminator.to(device)
        self.crit_GAN.to(device)
        self.crit_Feat.to(device)
        self.crit_Vgg.to(device)
        super().to(device)
        return self

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred_img(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake_1 = []
            fake_2 = []
            real = []
            for p in pred:
                fake_1.append([tensor[:tensor.size(0) // 3] for tensor in p])
                fake_2.append([tensor[
                    tensor.size(0) // 3:(tensor.size(0) // 3) * 2
                ] for tensor in p])
                real.append([tensor[(tensor.size(0) // 3) * 2:] for tensor in p])
        else:
            fake_1 = pred[:pred.size(0) // 3]
            fake_2 = pred[pred.size(0) // 3: (pred.size(0) // 3) * 2]
            real = pred[(pred.size(0) // 3) * 2:]

        return fake_1, fake_2, real

    def forward(self, pred, gts, update='generator'):
        pred_e = pred['edge']
        pred_I_d = pred['image_init']
        pred_I_dd = pred['image']

        gt_img = gts['rgb']
        gt_edge = gts['edge']
        gt_sem = gts['sem']

        if update == 'generator':
            G_losses = {}
            img_G_losses = {}
            # Edge GAN Loss
            fake_concat = torch.cat([gt_sem, pred_e], dim=1)
            real_concat = torch.cat([gt_sem, gt_edge], dim=1)

            # In Batch Normalization, the fake and real images are
            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

            discriminator_out = self.edge_discriminator(fake_and_real)

            pred_fake, pred_real = self.divide_pred(discriminator_out)

            G_loss = self.crit_GAN(pred_fake, target_is_real=True,
                                   for_discriminator=False)
            G_losses['GAN'] = G_loss

            num_D = len(pred_fake)
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.crit_Feat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

            G_losses['VGG'] = self.crit_Vgg(
                pred_e.expand(-1, 3, -1, -1),
                gt_edge.expand(-1, 3, -1, -1)) * self.lambda_vgg

            # Image GAN Loss
            img_real_concat = torch.cat([gt_sem, gt_img], dim=1)
            img_fake_concat_1 = torch.cat([gt_sem, pred_I_d], dim=1)
            img_fake_concat_2 = torch.cat([gt_sem, pred_I_dd], dim=1)
            img_fake_and_real = torch.cat([img_fake_concat_1,
                                           img_fake_concat_2, img_real_concat],
                                          dim=0)
            img_discriminator_out = self.image_discriminator(img_fake_and_real)
            img_pred_fake_1, img_pred_fake_2, img_pred_real = self.divide_pred_img(img_discriminator_out)
            G_loss_1 = self.crit_GAN(img_pred_fake_1, target_is_real=True,
                                     for_discriminator=False)
            G_loss_2 = self.crit_GAN(img_pred_fake_2, target_is_real=True,
                                     for_discriminator=False)
            img_G_losses['GAN_1'] = G_loss_1
            img_G_losses['GAN_2'] = G_loss_2

            num_D = len(img_pred_fake_1)
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(img_pred_fake_1[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.crit_Feat(
                        img_pred_fake_1[i][j], img_pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.lambda_feat / num_D
            img_G_losses['GAN_Feat_1'] = GAN_Feat_loss

            img_G_losses['VGG_1'] = self.crit_Vgg(pred_I_d, gt_img) * self.lambda_vgg

            num_D = len(img_pred_fake_2)
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(img_pred_fake_2[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.crit_Feat(
                        img_pred_fake_2[i][j], img_pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.lambda_feat / num_D
            img_G_losses['GAN_Feat_2'] = GAN_Feat_loss

            img_G_losses['VGG_2'] = self.crit_Vgg(pred_I_dd, gt_img) * self.lambda_vgg

            total_loss =\
                    self.lambda_gan * G_losses['GAN'] + self.lambda_feat * G_losses['GAN_Feat'] + self.lambda_vgg * G_losses['VGG'] +\
                    self.lambda_gan * img_G_losses['GAN_1'] + self.lambda_feat * img_G_losses['GAN_Feat_1'] + self.lambda_vgg * img_G_losses['VGG_1'] +\
                    self.lambda_gan * img_G_losses['GAN_2'] + self.lambda_feat * img_G_losses['GAN_Feat_2'] + self.lambda_vgg * img_G_losses['VGG_2']

            ret_pack = [G_losses, img_G_losses]

        elif update == 'discriminator':
            D_losses = {}
            img_D_losses = {}
            # For Edge discimination
            fake_edges = pred_e.detach()
            fake_concat = torch.cat([gt_sem, fake_edges], dim=1)
            real_concat = torch.cat([gt_sem, gt_edge], dim=1)

            # In Batch Normalization, the fake and real images are

            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

            discriminator_out = self.edge_discriminator(fake_and_real)


            pred_fake, pred_real = self.divide_pred(discriminator_out)

            D_losses['D_Fake'] = self.crit_GAN(pred_fake, False,
                                               for_discriminator=True)
            D_losses['D_real'] = self.crit_GAN(pred_real, True,
                                               for_discriminator=True)

            # For Img discrimination
            fake_img_1 = pred_I_d.detach()
            fake_img_2 = pred_I_dd.detach()
            img_real_concat = torch.cat([gt_sem, gt_img], dim=1)
            img_fake_concat_1 = torch.cat([gt_sem, fake_img_1], dim=1)
            img_fake_concat_2 = torch.cat([gt_sem, fake_img_2], dim=1)
            img_fake_and_real = torch.cat([img_fake_concat_1,
                                           img_fake_concat_2, img_real_concat],
                                          dim=0)
            img_discriminator_out = self.image_discriminator(img_fake_and_real)
            img_pred_fake_1, img_pred_fake_2, img_pred_real = self.divide_pred_img(img_discriminator_out)

            img_D_losses['D_Fake_1'] = self.crit_GAN(img_pred_fake_1, False,
                                                     for_discriminator=True)
            img_D_losses['D_Fake_2'] = self.crit_GAN(img_pred_fake_2, False,
                                                     for_discriminator=True)
            img_D_losses['D_real'] = self.crit_GAN(pred_real, True,
                                                   for_discriminator=True)

            total_loss = D_losses['D_Fake'] + D_losses['D_real'] +\
                    img_D_losses['D_Fake_1'] + self.lambd * img_D_losses['D_Fake_2'] +\
                    (self.lambd + 1) * img_D_losses['D_real']
            ret_pack = [D_losses, img_D_losses]

        return total_loss.mean(), ret_pack


if __name__ == '__main__':
    from Network import printTensorList
    from architectures.discriminator import MultiscaleDiscriminator as discriminator
    cudaDevice = ''

    if len(cudaDevice) < 1:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('[*] GPU Device selected as default execution device.')
        else:
            device = torch.device('cpu')
            print('[X] WARN: No GPU Devices found on the system! Using the CPU.'
                  'Execution maybe slow!')
    else:
        device = torch.device('cuda:%s' % cudaDevice)
        print('[*] GPU Device %s selected as default execution device.' %
              cudaDevice)

    seg_classes = 34
    edge_d = discriminator(in_channels=seg_classes + 1).to(device)
    image_d = discriminator(in_channels=seg_classes + 3).to(device)
    d = MultiModalityDiscriminatorLoss(edge_discriminator=edge_d,
                                       image_discriminator=image_d).to(device)
    dummy_seg = torch.Tensor(4, seg_classes, 256, 512).to(device)
    dummy_img = torch.Tensor(4, 3, 256, 512).to(device)
    dummy_edge = torch.Tensor(4, 1, 256, 512).to(device)
    pred_pack = {
        'edge': dummy_edge,
        'image_init': dummy_img,
        'image': dummy_img
    }

    gt_pack = {
        'rgb': dummy_img,
        'edge': dummy_edge,
        'sem': dummy_seg
        }

    print('Calculating Generator')
    out, ret_pack = d(pred_pack, gt_pack)
    print(out)
    print('Backward')
    out.backward()
    print('Calculating Discriminator')
    out, ret_pack = d(pred_pack, gt_pack, update='discriminator')
    print(out)
    print('Backward')
    out.backward()
