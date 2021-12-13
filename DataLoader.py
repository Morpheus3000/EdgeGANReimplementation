from torch.utils.data import Dataset
import numpy as np
import imageio
import pathlib
import scipy.io as sio
import cv2
import pickle


class CityscapesDataset(Dataset):
    # Load from mat
    def __init__(self, data_file, prefix, resize=[512, 256], gray=True, transform=None):
        self.prefix = prefix
        self.data_paths = self._read_data_file(data_file)
        self.gray = gray
        self.resize = resize
#         self.transform = transform
    def _read_data_file(self, data_file_path):

        filer = open(data_file_path, 'r')
        lines = filer.readlines()
        lines = [x.strip() for x in lines]

        return lines

    def __len__(self):
        return len(self.data_paths)
        # return 10

    def __getitem__(self, index):
        file = self.data_paths[index]

        # TODO: map semantic label to 1 hot indexing
        # TODO: resize mapped semantic labels
        # TODO: Check edge value range and channels

        # Read segmentation image
        im = imageio.imread(self.prefix + '/segs/' + file)
        seg = im.astype(np.float32)
        seg[np.isnan(seg)] = 0

        # Read colour image
        im = imageio.imread(self.prefix + '/colour/' + file)
        rgb = im.astype(np.float32)
        rgb[np.isnan(rgb)] = 0
        rgb = cv2.resize(rgb, self.resize)
        rgb = rgb / 255

        rgb = rgb.transpose((2, 0, 1))

        # Read edge image
        im = imageio.imread(self.prefix + '/edge/' + file)
        edge = im.astype(np.float32)
        edge[np.isnan(edge)] = 0
        edge = cv2.resize(edge, self.resize)

        edge = edge.transpose((2, 0, 1))

        image_dict = {'rgb': rgb,
                      'edge': edge,
                      'sem': sem}
        return image_dict, self.data_paths[index]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    data_root = ''
    train_list = ''
    test_list = ''

    print('[+] Init dataloader')
    testSet = CityscapesDataset(test_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=4, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, (images, _) = next(enu)
        b = time.time()
        alb = images['albedo']
        shd = images['shading']
        print(alb.max(), alb.min(), alb.mean())
        print(shd.max(), shd.min(), shd.mean())
        print('[*] Time taken: ', b - a)
