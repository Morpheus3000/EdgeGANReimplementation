import os
import time

from tqdm import tqdm
import numpy as np
import imageio

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from Network import EdgeGuidedNetwork
from DataLoader import CityscapesDataset
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

ExperimentName = 'EdgeGANReimplementation'
iter_targ = 74200

modelLoc = './Models/%s/' % ExperimentName.replace(' ', '_')

outdir = modelLoc + 'OutputJpeg%s' % ExperimentName.replace(' ', '_')
os.makedirs(outdir, exist_ok=True)
model_targ = outdir + '/snapshot_%d.t7' % iter_targ

saveLoc = './Infer_%s/' % ExperimentName.replace(' ', '_')
infer_loc = saveLoc + 'Visuals/'
os.makedirs(infer_loc, exist_ok=True)

data_root = '/home/Datasets/Cityscapes/Cityscapes_train_full/'
test_list = '/home/Datasets/Cityscapes/test_list.str'

seg_classes = 34
batch_size = 1
nthreads = 48
if batch_size < nthreads:
    nthreads = batch_size

done = u'\u2713'

print('[I] STATUS: Create utils instances...', end='')
support = mor_utils(device)
print(done)
print('[I] STATUS: Initiate Networks and transfer to device...', end='')

net = EdgeGuidedNetwork(seg_classes).to(device)

model, _, _ = support.loadModels(
    {'G': net,},
    model_targ
)

net = model['G']

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!...", end='')
    net = nn.DataParallel(net)
net.to(device)

print(done)

print('[I] STATUS: Initiate Dataloaders...')
testset = CityscapesDataset(test_list, data_root)

testLoader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                         num_workers=nthreads, pin_memory=True, drop_last=False)
batches_test = len(testLoader)
samples_test = len(testLoader.dataset)
print('\t[*] Train set with %d samples and %d batches.' % (samples_test,
                                                           batches_test),
      end='')
print(done)

def Infer(net):
    net.eval()
    t = tqdm(enumerate(testLoader), total=batches_test, leave=True)

    for i, data in t:
        images, file_name = data

        seg = Variable(images['sem']).to(device)
        rgb = Variable(images['rgb']).to(device)
        edge = Variable(images['edge']).to(device)
        edge = edge.expand(-1, 3, -1, -1)

        with torch.no_grad():
            pred = net(seg)

        pred_rgb = pred['image']

        b, _, _, _ = pred_rgb.shape

        for j in range(b):
            n = file_name[j]
            img = pred_rgb[j, :, :, :].cpu().detach().clone().numpy()
            img = img - img.min()
            img = img / img.max()
            img = img * 255
            img = img.transpose((1, 2, 0))
            img = img.astype(np.uint8)
            imageio.imwrite(infer_loc + '/%s.png' % n, img)


if __name__ == '__main__':
    print('[*] Beginning Training:')
    print('\tModels loaded from: ', outdir)
    print('\tModel infer iter: ', iter_targ)
    print('\tExperiment Name: ', ExperimentName)
    print('\tVisuals saved at: ', infer_loc)

    Infer(net)
