import os
import time

from tqdm import tqdm
import numpy as np
import imageio

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Network import EdgeGuidedNetwork
from DataLoader import CityscapesDataset
# from Criterions import 
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

visuals = 'Visuals%s' % ExperimentName.replace(' ', '_')
os.makedirs(visuals, exist_ok=True)

modelSaveLoc = outdir + '/snapshot_%d.t7'
logTrain = '%s/LogTrain.str' % outdir
logTest = '%s/LogTest.str' % outdir

data_root = '/ssdstore///v4/Processed/Intrinsics/'
train_list = '/home//Experiments//v4/train_files.txt'
test_list = '/home//Experiments//v4/test_files.txt'

seg_classes = 30
batch_size = 4
nthreads = 4
if batch_size < nthreads:
    nthreads = batch_size
max_epochs = 200 # 250
displayIter = 10
saveIter = 50000

# learningRate = 0.01
learningRate = 2e-4
# learningRate = 2e-4
beta = 0.5
# weightDecay = 1e-5

done = u'\u2713'

print('[I] STATUS: Create utils instances...', end='')
support = mor_utils(device)
print(done)
print('[I] STATUS: Initiate Network and transfer to device...', end='')

net = EdgeGuidedNetwork(seg_classes).to(device)
net.init_weights()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!...", end='')
    net = nn.DataParallel(net)
net.to(device)

print(done)
print('[I] STATUS: Initiate optimizer...', end='')
# optimizer = torch.optim.Adadelta(net.parameters(), lr=learningRate,
#                                  weight_decay=weightDecay)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, betas=(beta,
                                                                       0.999))

print(done)
print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
criterion = ScaleInvMSEScaleClampedIllumEdge(0.95).to(device)
dssim = SSIM(7, reduction='mean').to(device)
percep = VGGPerceptualLoss().to(device)

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

    loss_albedo = np.empty(batches_train)
    loss_unrefined_albedo = np.empty(batches_train)
    loss_unrefined_shading = np.empty(batches_train)
    loss_shading = np.empty(batches_train)
    loss_pred_recon = np.empty(batches_train)

    loss_pred_edge = np.empty(batches_train)
    loss_pred_edge_64 = np.empty(batches_train)
    loss_pred_edge_128 = np.empty(batches_train)

    # Perception on albedo 0.05
    loss_percep_albedo = np.empty(batches_train)

    # DSSIM on albedo 0.5
    loss_dssim_albedo = np.empty(batches_train)

    # DSSIM on shading 0.5
    loss_dssim_shading = np.empty(batches_train)

    loss_albedo[:] = np.nan
    loss_unrefined_albedo[:] = np.nan
    loss_unrefined_shading[:] = np.nan
    loss_shading[:] = np.nan
    loss_pred_recon[:] = np.nan

    loss_pred_edge[:] = np.nan
    loss_pred_edge_64[:] = np.nan
    loss_pred_edge_128[:] = np.nan

    loss_percep_albedo[:] = np.nan
    loss_dssim_albedo[:] = np.nan
    loss_dssim_shading[:] = np.nan

    Epoch_time = time.time()

    for i, data in t:
        iter_count += 1
        images, _ = data

        # rgb = Variable(images[0]).to(device)
        alb = Variable(images['albedo']).to(device)
        shd = Variable(images['shading']).to(device)
        deln = Variable(images['deln']).to(device)
        # mask = Variable(torch.ones(batch_size, 256, 256)).to(device)
        mask = Variable(images['mask']).to(device)
        shd = shd.unsqueeze(1)
        deln[deln <= 0.3] = 0
        [b, c, w, h] = shd.shape
        _, _, _, alb_edges, _, _ = edger(alb)
        _, _, _, illum_edges, _, _ = edger(deln)

        alb_edges = alb_edges.detach()
        illum_edges = illum_edges.detach()

        # illum_edges = shd_edges - alb_edges

        alb_edges_64 = F.interpolate(alb_edges, size=64)
        alb_edges_128 = F.interpolate(alb_edges, size=128)

        alb_edges_64 = alb_edges_64.detach()
        alb_edges_128 = alb_edges_128.detach()

        # edges = edges.to(device)

        illum_edges_64 = F.interpolate(illum_edges, size=64)
        illum_edges_128 = F.interpolate(illum_edges, size=128)

        illum_edges_64 = illum_edges_64.detach()
        illum_edges_128 = illum_edges_128.detach()

        # Expand to save copies
        # shd = shd.unsqueeze(1).expand(b, 3, w, h)
        # mask = mask.unsqueeze(1).expand(b, 3, w, h)

        rgb = torch.mul(alb, shd.expand(b, 3, w, h))

        optimizer.zero_grad()

        net_time = time.time()
        pred = net(rgb)

        # Pred gt dictionary for the criterion
        targets = {'reflectance': alb,
                   'shading': shd,
                   'reflec_edge': alb_edges,
                   'reflec_edge_64': alb_edges_64,
                   'reflec_edge_128': alb_edges_128,
                   'illum_edge': illum_edges,
                   'illum_edge_64': illum_edges_64,
                   'illum_edge_128': illum_edges_128,
                   'rgb': rgb}

        loss = criterion(pred, targets, mask)
        loss_alb = loss['reflectance']
        loss_un_alb = loss['unrefined_reflec']
        loss_un_shd = loss['unrefined_shd']
        loss_shd = loss['shading']
        loss_recon = loss['recon']
        loss_reflec_edge = loss['reflec_edge']
        loss_reflec_edge_64 = loss['reflec_edge_64']
        loss_reflec_edge_128 = loss['reflec_edge_128']
        loss_illum_edge = loss['illum_edge']
        loss_illum_edge_64 = loss['illum_edge_64']
        loss_illum_edge_128 = loss['illum_edge_128']

        # Percep Loss
        alb_percep = percep(pred['reflectance'], targets['reflectance'])

        # DSSIM Loss
        alb_dssim = dssim(pred['reflectance'], targets['reflectance'])
        shd_dssim = dssim(pred['shading'], targets['shading'])

        total_reflec_edge_loss = loss_reflec_edge + loss_reflec_edge_64 + loss_reflec_edge_128
        total_illum_edge_loss = loss_illum_edge + loss_illum_edge_64 + loss_illum_edge_128
        total_edge_loss = total_reflec_edge_loss + total_illum_edge_loss

        total_unrefined_loss = loss_un_alb + loss_un_shd

        total_loss = loss_alb + loss_shd + loss_recon + (0.5 *
                                                         total_unrefined_loss) +\
                0.4 * total_edge_loss + 0.05 * alb_percep +\
                0.4 * alb_dssim +  0.4 * shd_dssim

        total_loss.backward()
        optimizer.step()
        net_timed = time.time() - net_time

        loss_albedo[i] = loss_alb.cpu().detach().numpy()
        loss_unrefined_albedo[i] = loss_un_alb.cpu().detach().numpy()
        loss_unrefined_shading[i] = loss_un_shd.cpu().detach().numpy()
        loss_shading[i] = loss_shd.cpu().detach().numpy()
        loss_pred_recon[i] = loss_recon.cpu().detach().numpy()
        loss_pred_edge[i] = loss_reflec_edge.cpu().detach().numpy()
        loss_pred_edge_64[i] = loss_reflec_edge_64.cpu().detach().numpy()
        loss_pred_edge_128[i] = loss_reflec_edge_128.cpu().detach().numpy()
        loss_percep_albedo[i] = alb_percep.cpu().detach().numpy()
        loss_dssim_albedo[i] = alb_dssim.cpu().detach().numpy()
        loss_dssim_shading[i] = shd_dssim.cpu().detach().numpy()

        if iter_count % saveIter == 0:
            for j in range(batch_size):
                support.dumpOutputs(visuals, [pred['reflectance'][j, :, :, :] * mask[j, :, :, :],
                             pred['shading'][j, :, :, :].expand(3, w, h) *\
                             mask[j, :, :, :],
                             pred['reflec_edge'][j, :, :, :]],
                            gts=[rgb[j, :, :, :], alb[j, :, :, :],
                             shd[j, :, :, :].expand(3, w, h)], num=j + 1,
                                   iteration=iter_count)
            support.saveModels(net, optimizer, iter_count, modelSaveLoc % iter_count)
            tqdm.write('[!] Model Saved!')

        if iter_count % displayIter == 0:
            trainLogger.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d\n' % (loss_albedo[i],
                                                            loss_unrefined_albedo[i],
                                                            loss_shading[i],
                                                            loss_pred_recon[i],
                                                            loss_pred_edge[i],
                                                            loss_pred_edge_64[i],
                                                            loss_pred_edge_128[i],
                                                            loss_percep_albedo[i],
                                                            loss_dssim_albedo[i],
                                                            loss_dssim_shading[i],
                                                            iter_count))

        t.set_description('[Iter %d] Edge: %0.4f, Unref Reflec: %0.4f, Unref '
                          'Shd: %0.4f, Reflec:'
                          ' %0.4f, Shd: %0.4f, Recon: %0.4f, Percep: %0.4f,'
                          ' DSSIM-Alb: %0.4f, DSSIM-Shd: %0.4f, Epoch: %d, Time: '
                          '%0.4f' % (iter_count, loss_pred_edge[i],
                                     # loss_pred_edge_64[i],
                                     # loss_pred_edge_128[i],
                           loss_unrefined_albedo[i],
                           loss_unrefined_shading[i],
                           loss_albedo[i],
                           loss_shading[i],
                           loss_pred_recon[i],
                           0.05 * loss_percep_albedo[i],
                           0.5 * loss_dssim_albedo[i],
                           0.5 * loss_dssim_shading[i],
                           epoch_count, net_timed))


    avg_loss_albedo, avg_loss_shading = np.nanmean(loss_albedo),\
                                        np.nanmean(loss_shading)

    avg_loss_unref_albedo, avg_loss_edge = np.nanmean(loss_unrefined_albedo),\
                                        np.nanmean(loss_pred_edge)
    avg_loss_unref_shading = np.nanmean(loss_unrefined_shading)

    avg_loss_edge_64, avg_loss_edge_128  = np.nanmean(loss_pred_edge_64),\
                                        np.nanmean(loss_pred_edge_128)

    avg_loss_recon = np.nanmean(loss_pred_recon)

    avg_loss_percep = np.nanmean(loss_percep_albedo)
    avg_loss_dssim_alb = np.nanmean(loss_dssim_albedo)
    avg_loss_dssim_shd = np.nanmean(loss_dssim_shading)

    Epoch_timed = time.time() - Epoch_time
    print('[I] STATUS: Exp: %s: Epoch %d trained! Time taken: %0.2f minutes' %
                                                    (ExperimentName, epoch_count,
                                                               Epoch_timed / 60))

    return avg_loss_albedo, avg_loss_shading, avg_loss_recon,\
            avg_loss_unref_albedo, avg_loss_edge, avg_loss_percep,\
            avg_loss_dssim_alb, avg_loss_dssim_shd, avg_loss_edge_64,\
            avg_loss_edge_128, avg_loss_unref_shading

iter_count = 0
# print('[I] STATUS: Starting Pre-flight sub-systems checks!')
# 
# support.run_preflight_tests(net, {'inputs': torch.Tensor(batch_size, 3, 256,
#                                                          256).to(device),
#                                   'dest': visuals},
#                             {'optimizer': optimizer,
#                              'dest': outdir})

print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tSaving iter: ', saveIter)
print('\tModels Dumped at: ', outdir)
print('\tVisuals Dumped at: ', visuals)
print('\tExperiment Name: ', ExperimentName)

for i in range(max_epochs):
    avg_alb_loss, avg_shd_loss, avg_loss_recon, avg_loss_unref_albedo,\
            avg_loss_edge, avg_loss_percep, avg_loss_dssim_alb,\
            avg_loss_dssim_shd, avg_loss_edge_64, avg_loss_edge_128,\
    avg_loss_unref_shading = Train(net, i)
    print('[*] Epoch: %d: Edge: %0.4f, Unrefined Reflec: %0.4f, Unrefined Shd:'
          ' %0.4f, Reflec: %0.4f,'
          ' Shading: %0.4f, Recon: %0.4f, Perception: %0.4f, DSSIM-Alb: %0.4f,'
          ' DSSIM-Shd: %0.4f' % (i + 1, avg_loss_edge,
                                             # avg_loss_edge_64,
                                             # avg_loss_edge_128,
                                             avg_loss_unref_albedo,
                                             avg_loss_unref_shading,
                                             avg_alb_loss,
                                             avg_shd_loss,
                                             avg_loss_recon,
                                             avg_loss_percep,
                                             avg_loss_dssim_alb,
                                             avg_loss_dssim_shd,
                              ))
support.saveModels(net, optimizer, iter_count, modelSaveLoc % iter_count)
trainLogger.close()
testLogger.close()



