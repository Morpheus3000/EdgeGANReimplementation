import os
import time

from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from Network import EdgeGuidedNetwork
from DataLoader import CityscapesDataset
from Criterions import MultiModalityDiscriminatorLoss as crit
from Utils import mor_utils
from architectures.discriminator import MultiscaleDiscriminator as discriminator


torch.backends.cudnn.benchmark = True


cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

ExperimentName = 'EdgeGANReimplementation'

saveLoc = '/var/scratch/pdas/Experiments/%s/' % ExperimentName.replace(' ', '_')

outdir = saveLoc + 'OutputJpeg%s' % ExperimentName.replace(' ', '_')

modelSaveLoc = outdir + '/snapshot_%d.t7'

data_root = '/var/scratch/pdas/Datasets/Cityscapes/Cityscapes_train_full/'
train_list = '/var/scratch/pdas/Datasets/Cityscapes/train_list.str'
# test_list = '/home//Experiments//v4/test_files.txt'

seg_classes = 34
batch_size = 8
nthreads = 48
if batch_size < nthreads:
    nthreads = batch_size
max_epochs = 200 # 250
lr_mod_epoch = 100
displayIter = 10
# saveIter = 50000

lambda_gan = 1
lambda_feat = 10
lambda_vgg = 10
lambd = 2

learningRate = 2e-4
beta_1 = 0.5
beta_2 = 0.999

done = u'\u2713'

print('[I] STATUS: Create utils instances...', end='')
support = mor_utils(device)
print(done)
print('[I] STATUS: Initiate Networks and transfer to device...', end='')

net = EdgeGuidedNetwork(seg_classes).to(device)
net.init_weights('xavier', 0.02)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!...", end='')
    net = nn.DataParallel(net)
net.to(device)

discrim = discriminator(in_channels=(seg_classes + 3)).to(device)
discrim.init_weights('xavier', 0.02)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!...", end='')
    discrim = nn.DataParallel(discrim)
discrim.to(device)


print(done)
print('[I] STATUS: Initiate optimizer...', end='')
optimizer_G = torch.optim.Adam(net.parameters(), lr=learningRate / 2, betas=(beta_1, beta_2))
optimizer_D = torch.optim.Adam(discrim.parameters(), lr=learningRate / 2, betas=(beta_1, beta_2))
scheduler_G = torch.optim.lr_scheduler.LinearLR(optimizer_G, start_factor=1,
                                              end_factor=0, total_iters=100,
                                              verbose=True)
scheduler_D = torch.optim.lr_scheduler.LinearLR(optimizer_D, start_factor=1,
                                              end_factor=0, total_iters=100,
                                              verbose=True)

print(done)
print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
criterion = crit(
    discrim,
    seg_classes=seg_classes, lambda_feat=lambda_feat, lambda_gan=lambda_gan,
    lambda_vgg=lambda_vgg, lambd=lambd
).to(device)

print(done)
print('[I] STATUS: Initiate Dataloaders...')
trainset = CityscapesDataset(train_list, data_root)
# testset = CityscapesDataset(test_list, data_root)

trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                         num_workers=nthreads, pin_memory=True, drop_last=True)
batches_train = len(trainLoader)
samples_train = len(trainLoader.dataset)
print('\t[*] Train set with %d samples and %d batches.' % (samples_train,
                                                           batches_train),
      end='')
print(done)
# testLoader = DataLoader(testset, batch_size=batch_size, shuffle=False,
#                          num_workers=nthreads, pin_memory=True)
# batches_test = len(testLoader)
# samples_test = len(testLoader.dataset)
# print('\t[*] Test set with %d samples and %d batches.' % (samples_test,
#                                                           batches_test),
#       end='')
# print(done)


global iter_count


def Train(net, epoch_count):
    global iter_count
    net.train()
    epoch_count += 1
    t = tqdm(enumerate(trainLoader), total=batches_train, leave=False)

    edge_loss_G = np.empty(batches_train)
    edge_loss_D = np.empty(batches_train)
    edge_loss_feat = np.empty(batches_train)
    edge_loss_percep = np.empty(batches_train)

    img_d_loss_G = np.empty(batches_train)
    img_loss_D = np.empty(batches_train)
    img_d_loss_feat = np.empty(batches_train)
    img_d_loss_percep = np.empty(batches_train)

    img_dd_loss_G = np.empty(batches_train)
    img_dd_loss_feat = np.empty(batches_train)
    img_dd_loss_percep = np.empty(batches_train)

    edge_loss_G[:] = np.nan
    edge_loss_D[:] = np.nan
    edge_loss_feat[:] = np.nan
    edge_loss_percep[:] = np.nan

    img_d_loss_G[:] = np.nan
    img_loss_D[:] = np.nan
    img_d_loss_feat[:] = np.nan
    img_d_loss_percep[:] = np.nan

    img_dd_loss_G[:] = np.nan
    img_dd_loss_feat[:] = np.nan
    img_dd_loss_percep[:] = np.nan

    Epoch_time = time.time()

    for i, data in t:
        iter_count += 1
        images, _ = data

        # rgb = Variable(images[0]).to(device)
        seg = Variable(images['sem']).to(device)
        rgb = Variable(images['rgb']).to(device)
        edge = Variable(images['edge']).to(device)
        edge = edge.expand(-1, 3, -1, -1)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        net_time = time.time()
        pred = net(seg)

        target = {
            'sem': seg,
            'rgb': rgb,
            'edge': edge
        }

        # Update Generator

        total_loss_G, ret_pack_G = criterion(pred, target, update='generator')
        G_loss = ret_pack_G[0]
        img_G_loss = ret_pack_G[1]

        total_loss_G.backward()
        optimizer_G.step()

        edge_loss_G[i] = lambda_gan * G_loss['GAN'].cpu().detach().numpy()
        edge_loss_feat[i] = lambda_feat * G_loss['GAN_Feat'].cpu().detach().numpy()
        edge_loss_percep[i] = lambda_vgg * G_loss['VGG'].cpu().detach().numpy()

        img_d_loss_G[i] = lambda_gan * img_G_loss['GAN_1'].cpu().detach().numpy()
        img_d_loss_feat[i] = lambda_feat * img_G_loss['GAN_Feat_1'].cpu().detach().numpy()
        img_d_loss_percep[i] = lambda_vgg * img_G_loss['VGG_1'].cpu().detach().numpy()

        img_dd_loss_G[i] = lambda_gan * img_G_loss['GAN_2'].cpu().detach().numpy()
        img_dd_loss_feat[i] = lambda_feat * img_G_loss['GAN_Feat_2'].cpu().detach().numpy()
        img_dd_loss_percep[i] = lambda_vgg * img_G_loss['VGG_2'].cpu().detach().numpy()

        # Update Discriminator
        total_loss_D, ret_pack_D = criterion(pred, target, update='discriminator')
        D_loss = ret_pack_D[0]
        img_D_loss = ret_pack_D[1]

        total_loss_D.backward()
        optimizer_D.step()

        edge_loss_D[i] = D_loss['D_Fake'].cpu().detach().numpy() +\
            D_loss['D_real'].cpu().detach().numpy()
        img_loss_D[i] = (lambd + 1) * img_D_loss['D_real'].cpu().detach().numpy() +\
            img_D_loss['D_Fake_1'].cpu().detach().numpy() +\
            lambd * img_D_loss['D_Fake_2'].cpu().detach().numpy()

        net_timed = time.time() - net_time

        # if iter_count % saveIter == 0:
        t.set_description('[Iter %d] Feat_e: %0.4f, Percep_e: %0.4f,'
                          ' Feat_i_dd: %0.4f, Percep_i_dd: %0.4f,'
                          ' Epoch: %d, Time: %0.4f' % (
                              iter_count,
                              edge_loss_feat[i],
                              edge_loss_percep[i],

                              img_dd_loss_feat[i],
                              img_dd_loss_percep[i],

                              epoch_count, net_timed
                          ))

    avg_g_e_loss = np.nanmean(edge_loss_G)
    avg_f_e_loss = np.nanmean(edge_loss_feat)
    avg_p_e_loss = np.nanmean(edge_loss_percep)

    avg_g_d_loss = np.nanmean(img_d_loss_G)
    avg_f_d_loss = np.nanmean(img_d_loss_feat)
    avg_p_d_loss = np.nanmean(img_d_loss_percep)

    avg_g_dd_loss = np.nanmean(img_dd_loss_G)
    avg_f_dd_loss = np.nanmean(img_dd_loss_feat)
    avg_p_dd_loss = np.nanmean(img_dd_loss_percep)

    avg_e_D_loss = np.nanmean(edge_loss_D)
    avg_i_D_loss = np.nanmean(img_loss_D)

    Epoch_timed = time.time() - Epoch_time

    support.saveModels(
        {'G': net,
         'D': discrim},
        {'G': optimizer_G,
         'D': optimizer_D},
        iter_count,
        modelSaveLoc % iter_count,
        dd = torch.cuda.device_count() > 1)

    print('[!] Model Saved!')

    print('[I] STATUS: Exp: %s: Epoch %d trained! Time taken: %0.2f minutes' %
          (ExperimentName, epoch_count, Epoch_timed / 60))

    if epoch_count >= lr_mod_epoch - 1:
        scheduler_G.step()
        scheduler_D.step()

    return avg_g_e_loss,\
        avg_f_e_loss,\
        avg_p_e_loss,\
        avg_g_d_loss,\
        avg_f_d_loss,\
        avg_p_d_loss,\
        avg_g_dd_loss,\
        avg_f_dd_loss,\
        avg_p_dd_loss,\
        avg_e_D_loss,\
        avg_i_D_loss


iter_count = 0

print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tModels Dumped at: ', outdir)
print('\tExperiment Name: ', ExperimentName)
mod = {'G': net,
         'D': discrim}
support.saveModels(
        {'G': net,
         'D': discrim},
        {'G': optimizer_G,
         'D': optimizer_D},
        iter_count,
        modelSaveLoc % iter_count,
        dd = torch.cuda.device_count() > 1)
