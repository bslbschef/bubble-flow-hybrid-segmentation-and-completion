from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BubbleDataset(Dataset):
    def __init__(self, args, mask_root=None, image_root=None, func=None, transform=True, cropping=True):
        self.mask_root = mask_root
        self.image_root = image_root
        self.transform = transform
        self.split = [0.99, 0.01] # validation data is only used for output training samples
        self.seed = 213
        self.args = args
        self.func = func
        self.cropping = cropping
        np.random.seed(self.seed)

        self.masks = os.listdir(mask_root)
        self.images = os.listdir(image_root)

        self.mask_len = len(self.masks)
        self.images_len = len(self.images)


        train_images, val_images = self.images[:int(self.split[0] * self.images_len)], self.images[int(
            self.split[0] * self.images_len):]
        train_masks, val_masks = self.masks[:int(self.split[0] * self.mask_len)], self.masks[int(
            self.split[0] * self.mask_len):]

        if self.func == 'train':
            self.images = train_images
            self.masks = train_masks
        elif self.func == 'validation':
            self.images = val_images
            self.masks = val_masks
        elif self.func == 'testing':
            self.images = self.images
            self.masks = self.masks


        self.mask_len = len(self.masks)
        self.images_len = len(self.images)
        self.len_dataset = max(self.mask_len,self.images_len)


    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        image_dir = os.path.join(self.image_root,self.images[idx % self.images_len])
        mask_dir = os.path.join(self.mask_root,self.masks[idx % self.mask_len])

        if self.args.image_channel == 1:
            image_type = 'L'
        elif self.args.image_channel == 3:
            image_type = 'RGB'

        if self.args.mask_channel == 1:
            mask_type = 'L'
        elif self.args.mask_channel == 3:
            mask_type = 'RGB'


        image = np.array(Image.open(image_dir).convert(image_type))
        mask = np.array(Image.open(mask_dir).convert(mask_type))


        if self.cropping:
            cropping = A.Compose(
                [
                    A.RandomCrop(width=self.args.crop_size,height=self.args.crop_size)
                ],
                additional_targets={'mask' : 'mask'}
            )
            Cropping = cropping(image=image, mask=mask)
            image = Cropping['image']
            mask = Cropping['mask']

        if self.transform:
            img_trans = A.Compose([
                A.Normalize(mean=[0.5] * self.args.image_channel, std=[0.5] * self.args.image_channel,
                            max_pixel_value=255),
                ToTensorV2(),
            ])

            mask_trans = A.Compose([
                A.Normalize(mean=[0.5] * self.args.mask_channel, std=[0.5] * self.args.mask_channel,
                            max_pixel_value=255),
                ToTensorV2(),
            ])
            transform = img_trans(image=image)
            image = transform['image']
            transform = mask_trans(image=mask)
            mask = transform['image']

        return image, mask