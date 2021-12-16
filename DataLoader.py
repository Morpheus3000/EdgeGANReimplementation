from torch.utils.data import Dataset
import numpy as np
import imageio
import scipy.io as sio
import cv2


class CityscapesDataset(Dataset):
    def __init__(self, data_file, prefix, resize=(512, 256), seg_var='seg_res',
                 gray=True):
        self.prefix = prefix
        self.data_paths = self._read_data_file(data_file)
        self.gray = gray
        self.resize = resize
        self.seg_var = seg_var

    def _read_data_file(self, data_file_path):

        filer = open(data_file_path, 'r')
        lines = filer.readlines()
        lines = [x.strip() for x in lines]

        return lines

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        file = self.data_paths[index]

        # Read segmentation image
        s = '_'.join(file.split('_')[:-1])
        seg = sio.loadmat(self.prefix + '/segs/' + s + '_gtFine_labelIds.mat')[self.seg_var]
        seg = seg.astype(np.float32)
        seg = cv2.resize(seg, self.resize, interpolation=cv2.INTER_NEAREST)
        seg = seg.transpose((2, 0, 1))

        # Read colour image
        im = imageio.imread(self.prefix + '/colour/' + file + '.png')
        rgb = im.astype(np.float32)
        rgb[np.isnan(rgb)] = 0
        rgb = cv2.resize(rgb, self.resize)
        rgb = rgb / 255

        rgb = rgb.transpose((2, 0, 1))

        # Read edge image
        im = imageio.imread(self.prefix + '/edge/' + file + '.png')
        edge = im.astype(np.float32)
        edge[np.isnan(edge)] = 0
        edge = cv2.resize(edge, self.resize)

        image_dict = {'rgb': rgb,
                      'edge': edge,
                      'sem': seg}
        return image_dict, self.data_paths[index]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    from Network import printTensorList

    data_root = 'D:/Datasets/CityScapes/Cityscapes_train_full/'
    train_list = 'D:/Datasets/CityScapes/Cityscapes_train_full/train_list.str'

    print('[+] Init dataloader')
    testSet = CityscapesDataset(train_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=4, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(5):
        a = time.time()
        i, (images, _) = next(enu)
        b = time.time()
        printTensorList(images)
        print('[*] Time taken: ', b - a)
