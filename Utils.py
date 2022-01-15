import os
import imageio
import numpy as np
import torch
import torch.nn as nn


class mor_utils:

    def __init__(self, device):
        self.device = device

    def printTensorList(self, data):
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

    def saveModels(self, model, optims, iterations, path, dd=True):
        if dd:
            checkpoint = {
                'iters': iterations,
                'model': [
                    model['G'].module.state_dict(),
                    model['D'].module.state_dict(),
                ],
                'optimizer': [
                    optims['G'].state_dict(),
                    optims['D'].state_dict(),
                ]
            }
        else:
            checkpoint = {
                'iters': iterations,
                'model': [
                    model['G'].state_dict(),
                    model['D'].state_dict(),
                ],
                'optimizer': [
                    optims['G'].state_dict(),
                    optims['D'].state_dict(),
                ]
            }
        torch.save(checkpoint, path)

    def loadModels(self, model, path, optims=None, Test=True,
                   load_discrim=False):
        checkpoint = torch.load(path)
        model['G'].load_state_dict(checkpoint['model'][0])
        if load_discrim:
            model['D'].load_state_dict(checkpoint['model'][1])
        if not Test:
            optims['G'].load_state_dict(checkpoint['optimizer'][0])
            optims['D'].load_state_dict(checkpoint['optimizer'][1])
        return model, optims, checkpoint['iters']
