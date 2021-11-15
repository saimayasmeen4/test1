import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def save_train_image(img_path, scales, patch_size, stride, aug_times,train_num, h5f_file):
  img = cv2.imread(img_path)
  h, w, c = img.shape
  for k in range(len(scales)):
    Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
    Img = np.expand_dims(Img[:,:,0].copy(), 0)
    Img = np.float32(normalize(Img))
    # print("train_files: %s scale %.1f # samples: %d" % (train_files[i], scales[k], patches.shape[3]*aug_times))
            
    # dont use patches as input: use orignal scaled image as input
    h5f_file.create_dataset(str(train_num), data=Img)
    train_num += 1

    # # to save patches
    # patches = Im2Patch(Img, win=patch_size, stride=stride)
    # for n in range(patches.shape[3]):
    #   data = patches[:,:,:,n].copy()
    #   h5f_file.create_dataset(str(train_num), data=data)
    #   train_num += 1
    #   for m in range(aug_times-1):
    #     data_aug = data_augmentation(data, np.random.randint(1,8))
    #     h5f_file.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
    #     train_num += 1 

  return train_num 

def save_val_image(img_path, val_num, h5f_file):
  img = cv2.imread(img_path)
  img = np.expand_dims(img[:,:,0], 0)
  img = np.float32(normalize(img))
  h5f_file.create_dataset(str(val_num), data=img)
  val_num += 1

  return val_num


def prepare_data(data_path, patch_size, stride, aug_times=1, train_images=10, val_images=10):
    data_loader_path= "dataloader_files/"
    if not os.path.exists(data_loader_path):
        os.mkdir(data_loader_path)

    # train
    print('process training data')
    scales = [0.9]
    #scales = [1, 0.9, 0.8, 0.7]
    #clean_files = glob.glob(os.path.join('/content/drive/MyDrive/SAR/virtual_sar_training_set/clean_1c', '*'))
    clean_files = glob.glob(os.path.join('/content/drive/MyDrive/DudeNet2/gray/data/clean_1c', '*'))
    print('Total Clean files',len(clean_files)) 
    noisy_files = glob.glob(os.path.join('/content/drive/MyDrive/DudeNet2/gray/data/noisy1_c', '*'))
    print('Total Noisy files',len(noisy_files))

    if len(clean_files) != len(noisy_files):
      print("clean and noisy images are not same in length.")
      print("Please make sure clean and noisy images are same in length.")

    # sort
    clean_files.sort()
    noisy_files.sort()

    # set seed to read  
    # np.random.seed(101) 

    data_list = list(range(len(clean_files)))

    # reproducable shuffling
    random.Random(4).shuffle(data_list)

    train_indexes = data_list[0: train_images]
    val_indexes =  data_list[ (len(data_list) - val_images) : ]

    h5f_train_clean = h5py.File(data_loader_path+'train_clean.h5', 'w')
    h5f_train_noisy = h5py.File(data_loader_path+'train_noisy.h5', 'w')
    train_clean_num = 0
    train_noisy_num = 0
    for index in train_indexes:
        print('Reading train Image: ', clean_files[index])
        train_clean_num = save_train_image(clean_files[index], scales, patch_size, stride, aug_times, train_clean_num, h5f_train_clean)
        train_noisy_num = save_train_image(noisy_files[index], scales, patch_size, stride, aug_times, train_noisy_num, h5f_train_noisy)
    h5f_train_clean.close()
    h5f_train_noisy.close()

    # val
    print('\nprocess validation data')
    h5f_val_clean = h5py.File(data_loader_path+'val_clean.h5', 'w')
    h5f_val_noisy = h5py.File(data_loader_path+'val_noisy.h5', 'w')

    val_clean_num = 0
    val_noisy_num = 0
    for index in val_indexes:
        print('Reading val Image: ', clean_files[index])
        val_clean_num = save_val_image(clean_files[index], val_clean_num, h5f_val_clean)
        val_noisy_num = save_val_image(noisy_files[index], val_noisy_num, h5f_val_noisy)
    h5f_val_clean.close()
    h5f_val_noisy.close()

    print('training set, # samples %d\n' % train_clean_num)
    print('val set, # samples %d\n' % val_clean_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('dataloader_files/train_clean.h5', 'r')
        else:
            h5f = h5py.File('dataloader_files/val_clean.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f_clean = h5py.File('dataloader_files/train_clean.h5', 'r')
            h5f_noisy = h5py.File('dataloader_files/train_noisy.h5', 'r')
        else:
            h5f_clean = h5py.File('dataloader_files/val_clean.h5', 'r')
            h5f_noisy = h5py.File('dataloader_files/val_noisy.h5', 'r')
        key = self.keys[index]
        data_clean = np.array(h5f_clean[key])
        data_noisy = np.array(h5f_noisy[key])
        data = np.stack([ data_noisy, data_clean], axis=0)
        h5f_clean.close()
        h5f_noisy.close()
        return torch.Tensor(data)
