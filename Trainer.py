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
from Criterions import MultiModalityDiscriminator
from Utils import mor_utils

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

ExperimentName = 'EdgeGANVanilla'

outdir = 'OutputJpeg%s' % ExperimentName.replace(' ', '_')
os.makedirs(outdir, exist_ok=True)

modelSaveLoc = outdir + '/snapshot_%d.t7'
logTrain = '%s/LogTrain.str' % outdir
logTest = '%s/LogTest.str' % outdir

data_root = '/ssdstore///v4/Processed/Intrinsics/'
train_list = '/home//Experiments//v4/train_files.txt'
test_list = '/home//Experiments//v4/test_files.txt'

seg_classes = 34
batch_size = 4
nthreads = 4
if batch_size < nthreads:
    nthreads = batch_size
max_epochs = 200 # 250
lr_mod_epoch = 100
displayIter = 10
saveIter = 50000

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
print('[I] STATUS: Initiate Network and transfer to device...', end='')

net = EdgeGuidedNetwork(seg_classes).to(device)
net.init_weights('xavier', 0.02)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!...", end='')
    net = nn.DataParallel(net)
net.to(device)

print(done)
print('[I] STATUS: Initiate optimizer...', end='')
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate / 2, betas=(beta_1, beta_2))
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                              end_factor=0, total_iters=100,
                                              verbose=True)

print(done)
print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
criterion = MultiModalityDiscriminator(
    seg_classes=seg_classes, lambda_feat=lambda_feat, lambda_gan=lambda_gan,
    lambda_vgg=lambda_vgg, lambd=lambd
).to(device)

print(done)
print('[I] STATUS: Initiate Dataloaders...')
trainset = CityscapesDataset(train_list, data_root)
testset = CityscapesDataset(test_list, data_root)

trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                         num_workers=nthreads, pin_memory=True, drop_last=True)
batches_train = len(trainLoader)
samples_train = len(trainLoader.dataset)
print('\t[*] Train set with %d samples and %d batches.' % (samples_train,
                                                           batches_train),
      end='')
print(done)
testLoader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                         num_workers=nthreads, pin_memory=True)
batches_test = len(testLoader)
samples_test = len(testLoader.dataset)
print('\t[*] Test set with %d samples and %d batches.' % (samples_test,
                                                          batches_test),
      end='')
print(done)


print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
testLogger = open(logTest, 'w')
print(done)

multiplier = 1
blank = torch.zeros(batch_size, 1, 256, 256)

global iter_count

def Train(net, epoch_count):
    global iter_count
    net.train()
    epoch_count += 1
    t = tqdm(enumerate(trainLoader), total=batches_train, leave=False)

    loss_GAN = np.empty(batches_train)
    loss_feat = np.empty(batches_train)
    loss_percep = np.empty(batches_train)

    loss_GAN[:] = np.nan
    loss_feat[:] = np.nan
    loss_percep[:] = np.nan

    Epoch_time = time.time()

    for i, data in t:
        iter_count += 1
        images, _ = data

        # rgb = Variable(images[0]).to(device)
        seg = Variable(images['sem']).to(device)
        edge = Variable(images['edge']).to(device)
        rgb = Variable(images['rgb']).to(device)

        optimizer.zero_grad()

        net_time = time.time()
        pred = net(seg)

        loss = criterion(pred, images, mask)
        gan_loss = loss['gan']
        feat_loss = loss['feat']
        percep_loss = loss['percep']

        total_loss = lambda_c * gan_loss +\
                lambda_f * feat_loss +\
                lambda_p * percep_loss

        total_loss.backward()
        optimizer.step()
        net_timed = time.time() - net_time

        loss_GAN[i] = lambda_c * gan_loss.cpu().detach().numpy()
        loss_feat[i] = lambda_f * feat_loss.cpu().detach().numpy()
        loss_percep[i] = lambda_p * percep_loss.cpu().detach().numpy()

        if iter_count % saveIter == 0:
            support.saveModels(net, optimizer, iter_count, modelSaveLoc % iter_count)
            tqdm.write('[!] Model Saved!')

        t.set_description('[Iter %d] GAN: %0.4f, Feature: %0.4f, Percep: %0.4f,'
                          ' Epoch: %d, Time: %0.4f' % (
                              iter_count, loss_pred_edge[i],
                              loss_GAN[i],
                              loss_feat[i],
                              loss_percep[i],
                              epoch_count, net_timed
                          ))

    avg_gan_loss = np.nanmean(loss_GAN)
    avg_feat_loss = np.nanmean(loss_feat)
    avg_percep_loss = np.nanmean(loss_percep)

    Epoch_timed = time.time() - Epoch_time
    print('[I] STATUS: Exp: %s: Epoch %d trained! Time taken: %0.2f minutes' %
                                                    (ExperimentName, epoch_count,
                                                               Epoch_timed / 60))
    if epoch_count >= lr_mod_epoch - 1:
        scheduler.step()

    return avg_gan_loss, avg_feat_loss, avg_percep_loss

iter_count = 0

print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tSaving iter: ', saveIter)
print('\tModels Dumped at: ', outdir)
print('\tExperiment Name: ', ExperimentName)

for i in range(max_epochs):
    avg_gan_loss, avg_feat_loss, avg_percep_loss = Train(net, i)
    print('[*] Epoch: %d - GAN: %0.4f, Feature: %0.4f, Percep: %0.4f,' % (
        i + 1, avg_gan_loss, avg_feat_loss, avg_percep_loss
    ))
support.saveModels(net, optimizer, iter_count, modelSaveLoc % iter_count)
trainLogger.close()
testLogger.close()
