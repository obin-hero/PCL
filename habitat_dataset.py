import torch.utils.data as data
import numpy as np
import joblib
import torch
import time
import cv2
import quaternion as q
import os
from PIL import Image

class HabitatDataset(data.Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, img):
        data_patches = np.stack([img[:, i * 21:(i + 1) * 21] for i in range(12)])
        index_list = np.arange(0, 12).tolist()
        random_cut = np.random.randint(12)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_patches = data_patches[index_list]
        augmented_img = np.concatenate(np.split(permuted_patches, 12, axis=0), 2)[0]
        return augmented_img

    def pull_image(self, index):
        data = joblib.load(self.data_list[index])
        curr_im = np.concatenate([data['rgb'].astype(np.float32)/255.0, data['depth']],2)# * 2 - 1
        curr_im1 = self.augment(curr_im)
        curr_im2 = self.augment(curr_im)
        #curr_im = self.transform(Image.fromarray(curr_im))
        curr_im1 = torch.from_numpy(curr_im1.transpose(2, 0, 1))
        curr_im2 = torch.from_numpy(curr_im2.transpose(2, 0, 1))
        return (curr_im1, curr_im2), index

if __name__=='__main__':
    from default_cfg import get_config
    from torch.utils.data.dataloader import DataLoader
    cfg = get_config()
    split = 'train'
    DATA_DIR = '/media/obin/5d368da0-d601-490b-b5d8-6122946470b8/DATA/loc_data_12view/'
    train_data_list = [os.path.join(DATA_DIR, split, x) for x in sorted(os.listdir(os.path.join(DATA_DIR, split)))]
    train_dataset = HabitatMultipletDataset(cfg, train_data_list, DATA_DIR, mode=split)
    params = {'batch_size': 2,#cfg.training.batch_size,
              'shuffle': True,
              'num_workers': 0,#cfg.training.num_workers,
              'pin_memory': True}
    train_dataloader = DataLoader(train_dataset, **params)
    train_iter = iter(train_dataloader)
    for batch in train_iter:
        print('get batch')