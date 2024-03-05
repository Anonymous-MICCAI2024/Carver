import sys
sys.path.append('..')
from utils import hdf5_reader
from skimage.transform import resize
from torch.utils.data import Dataset
import torch
import numpy as np
import random



class Trunc_and_Normalize(object):
    '''
    truncate gray scale and normalize to [0,1]
    '''
    def __init__(self, scale):
        self.scale = scale
        assert len(self.scale) == 2, 'scale error'

    def __call__(self, sample):
        image = sample['image']
        
        # gray truncation
        image = image - self.scale[0]
        gray_range = self.scale[1] - self.scale[0]
        image[image < 0] = 0
        image[image > gray_range] = gray_range
        
        # normalization
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image / gray_range
        sample['image'] = image
        return sample


class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[:,crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None, num_class=2, crop=0):
        self.dim = dim
        self.num_class = num_class
        self.crop = crop

    def __call__(self, sample):

        # image: numpy array
        # label: numpy array
        image = sample['image']
        label = sample['label']
        # crop
        if self.crop != 0:
            if len(image.shape) > 2:
                image = image[:,self.crop:-self.crop, self.crop:-self.crop]
                label = label[:,self.crop:-self.crop, self.crop:-self.crop]
            else:
                image = image[self.crop:-self.crop, self.crop:-self.crop]
                label = label[self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and image.shape != self.dim:
            image = resize(image, self.dim, anti_aliasing=True)
            temp_label = np.zeros(self.dim,dtype=np.float32)
            for z in range(1, self.num_class):
                roi = resize((label == z).astype(np.float32),self.dim,mode='constant')
                temp_label[roi >= 0.5] = z
            label = temp_label

        sample['image'] = image
        sample['label'] = label
        return sample


class To_Tensor(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''
    def __init__(self,num_class=2):
        self.num_class = num_class

    def __call__(self,sample):

        image = sample['image']
        label = sample['label']
        
        # expand dims
        if len(image.shape) == 3:
            new_image = np.expand_dims(image,axis=0)
        else:
            new_image = image
        new_label = np.empty((self.num_class,) + label.shape, dtype=np.float32)
        for z in range(1, self.num_class):
            temp = (label==z).astype(np.float32)
            new_label[z,...] = temp
        new_label[0,...] = np.amax(new_label[1:,...],axis=0) == 0   
    
        # convert to Tensor
        sample['image'] = torch.from_numpy(new_image)
        sample['label'] = torch.from_numpy(new_label)
        return sample



class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args:
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self, path_list, roi_number=None, num_class=2, transform=None):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self,index):
        # Get image and label
        # image: (D,H,W) or (H,W) 
        # label: same shape with image, integer, [0,1,...,num_class-1]
        image = hdf5_reader(self.path_list[index],'image')
        image = image[0]
        label = hdf5_reader(self.path_list[index],'label')
        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(label,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i+1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label==self.roi_number).astype(np.float32) 
        # print(image.shape)
        sample = {'image':image, 'label':label}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)
        # print(sample['image'].shape)
        # print(sample['label'].shape)
        return sample



class BalanceDataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args:
    - path_list: list of two lists, one includes positive samples, and the other includes negative samples
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self,
                 path_list=None,
                 roi_number=None,
                 num_class=2,
                 transform=None,
                 factor=0.3):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform
        self.factor = factor


    def __len__(self):
        assert isinstance(self.path_list,list)
        assert len(self.path_list) == 2
        return sum([len(case) for case in self.path_list])

    def __getitem__(self, index):
        # balance sampler
        item_path = random.choice(self.path_list[int(random.random() < self.factor)])
        # Get image and mask
        image = hdf5_reader(item_path,'image')
        label = hdf5_reader(item_path,'label')

        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(label,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i+1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label == self.roi_number).astype(np.float32)

        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)


        return sample