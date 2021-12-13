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

    def saveModels(self, model, optims, iterations, path):
        # cpu = torch.device('cpu')
        if isinstance(model, nn.DataParallel):
            checkpoint = {
                'iters': iterations,
                'model': model.module.state_dict(),
                # 'model': model.module.to(cpu).state_dict(),
                'optimizer': optims.state_dict()
            }
        else:
            checkpoint = {
                'iters': iterations,
                'model': model.state_dict(),
                # 'model': model.to(cpu).state_dict(),
                'optimizer': optims.state_dict()
            }
        torch.save(checkpoint, path)
        # model.to(self.device)

    def loadModels(self, model, path, optims=None, Test=True):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        if not Test:
            optims.load_state_dict(checkpoint['optimizer'])
        return model, optims, checkpoint['iters']

    def dumpOutputs(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            imageio.imwrite((vis + '/%s_pred_alb.png') % filename, pred_a)
            imageio.imwrite((vis + '/%s_pred_shd.png') % filename, pred_s)

    def dumpOutputs2(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            pred_e = preds[2].cpu().detach().clone().numpy()
            pred_e[pred_e < 0] = 0
            pred_e = (pred_e / pred_e.max()) * 255
            pred_e = pred_e.transpose((1, 2, 0))
            pred_e = pred_e.astype(np.uint8)

            pred_u = preds[3].cpu().detach().clone().numpy()
            pred_u[pred_u < 0] = 0
            pred_u = (pred_u / pred_u.max()) * 255
            pred_u = pred_u.transpose((1, 2, 0))
            pred_u = pred_u.astype(np.uint8)

            imageio.imwrite((vis + '/%s_pred_alb.png') % filename, pred_a)
            imageio.imwrite((vis + '/%s_pred_shd.png') % filename, pred_s)
            imageio.imwrite((vis + '/%s_pred_edge.png') % filename, pred_e)
            imageio.imwrite((vis + '/%s_pred_unrefined.png') % filename, pred_u)

    def dumpOutputs3(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            for k, ele in preds.items():
                pred = ele.cpu().detach().clone().numpy()
                pred[pred < 0] = 0
                pred = (pred / pred.max()) * 255
                pred = pred.transpose((1, 2, 0))
                pred = pred.astype(np.uint8)
                imageio.imwrite((vis + '/%s_%s.png') % (filename, k), pred)

    def run_preflight_tests(self, model, dataset_dict, model_dict):
        done = u'\u2713'
        print('\t[I] STATUS: Test network forward systems...', end='')
        with torch.no_grad():
            pred = model(dataset_dict['inputs'])
        print(done)
        print('\t[I] STATUS: Sanity check on the network predictions')
        self.printTensorList(pred)
        _, _, w, h = dataset_dict['inputs'].shape
        print('\t[I] STATUS: Test image dump systems...', end='')
        self.dumpOutputs(dataset_dict['dest'], [pred['reflectance'][0, :, :, :],
                                                pred['shading'][0, :, :, :].expand(3, w, h),
                                                pred['edge'][0, :, :, :]],
                         gts=[dataset_dict['inputs'][0, :, :, :],
                              dataset_dict['inputs'][0, :, :, :],
                              dataset_dict['inputs'][0, :, :, :]],
                         num=13,
                         iteration=1337)
        print(done)

        print('\t[I] STATUS: Test model saving systems...', end='')
        self.saveModels(model, model_dict['optimizer'], 1337,
                        model_dict['dest'] + '/test_dump.t7')
        print(done)
        print('\t[I] STATUS: Cleaning up...', end='')
        os.remove(model_dict['dest'] + '/test_dump.t7')
        os.remove(dataset_dict['dest'] + '/Out_13_1337.png')
        print(done)
        print('[I] STATUS: All pre-flight tests passed! All essential'
              ' sub-systems are green!')
