import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import os

import copy
import torchvision.transforms as transforms

from PIL import Image

# add the sys path
# import sys
# sys.path.append('..')
# sys.path.append('.')

# from custom_transforms import *
transform_list_train = {'resize': {'size': [256, 256]}, 
                    'random_flip': {'lr': True, 'ud': True}, 
                    'random_rotate': {'range': [0, 359]}, 
                    # 'random_rotate': {'range': [-20, 20]}, # A smaller range is used for the general loader
                    ##
                    # 'random_scale_crop': {'range': [0.75, 1.25]}, 
                    'random_image_enhance': {'methods': ['contrast', 'sharpness', 'brightness']}, 
                    # 'random_dilation_erosion': {'kernel_range': [2, 5]}, 
                    'random_gaussian_blur': {'apply': True},
                    ##
                    'tonumpy': None, 
                    'normalize': {'mean': None, 'std': None}, 
                    'totensor': None}

# transform_list_train = {'resize': {'size': [256, 256]}, 
#                     'random_flip': {'lr': True, 'ud': True}, 
#                     # 'random_rotate': {'range': [0, 359]}, 
#                     'random_rotate': {'range': [-20, 20]}, # A smaller range is used for the general loader
#                     'random_scale_crop': {'range': [0.75, 1.25]}, 
#                     'random_image_enhance': {'methods': ['contrast', 'sharpness', 'brightness']}, 
#                     'random_dilation_erosion': {'kernel_range': [2, 5]}, 
#                     'tonumpy': None, 
#                     # 'normalize': {'mean': [0.485, 0.456, 0.406], 
#                     # 'std': [0.229, 0.224, 0.225]}, 
#                     'normalize': {'mean': None, 'std': None}, 
#                     'totensor': None}
transform_list_test = {'resize': {'size': [256, 256]},
                    'tonumpy': None, 
                    # 'normalize': {'mean': [0.485, 0.456, 0.406], 
                    # 'std': [0.229, 0.224, 0.225]}, 
                    'normalize': {'mean': None, 'std': None}, 
                    'totensor': None}

import logging
# class GeneralSegDataset(torch.utils.data.Dataset):
#     def __init__(self, args, 
#         transform=None,
#         ops_weak=None,
#         ops_strong=None,
#         label_flag=False,
#         test_flag=False):

#         self.transform = transform
#         self.ops_weak = ops_weak
#         self.ops_strong = ops_strong

#         assert bool(ops_weak) == bool(
#             ops_strong
#         ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

#         root = os.path.join(args.root_path, 'test' if test_flag else 'train')
#         image_root, gt_root = os.path.join(root, 'images'), os.path.join(root, 'masks')

#         self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
#         self.images = sorted(self.images)

#         if not test_flag:
#             labeled_name_txt = os.path.join(root, 'labeled.txt')
#             if os.path.exists(labeled_name_txt):
#                 with open(labeled_name_txt, 'r') as f:
#                     self.labeled_names = f.read().splitlines()
#             else:
#                 raise ValueError("labeled.txt does not exist in {}".format(root))
#             # print(self.labeled_names)
#             self.labeled_names = [os.path.join(image_root, f) for f in self.labeled_names]
#             self.unlabeled_images = [f for f in self.images if f not in self.labeled_names]

#             self.images = self.labeled_names if label_flag else self.unlabeled_images
        
#         # self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.bmp')]
#         self.gts = [os.path.join(gt_root, f.split('/')[-1]) for f in self.images]
#         self.gts = sorted(self.gts)
        
#         self.filter_files()
        
#         self.size = len(self.images)
#         transform_list_strong = transform_list_train if not test_flag else transform_list_test
#         transform_list_strong['resize']['size'] = args.patch_size
#         transform_list_weak = copy.deepcopy(transform_list_strong)

#         transform_list_weak.pop('random_scale_crop', None)
#         transform_list_weak.pop('random_image_enhance', None)
#         transform_list_weak.pop('random_dilation_erosion', None)

#         # if not args.flip:
#         # transform_list_strong.pop('random_flip', None)
#     # if not args.rotate:
#         # transform_list_strong.pop('random_rotate', None)
#     # if not args.scale_crop:
#         transform_list_strong.pop('random_scale_crop', None)
#     # if not args.img_enhance:
#         # transform_list_strong.pop('random_image_enhance', None)
#     # if not args.dilation_erosion:
#         transform_list_strong.pop('random_dilation_erosion', None)
            
#         if not test_flag:
#             print("Weak data augmentation used in {}:".format('test' if test_flag else 'train'))
#             print(transform_list_weak)
#             print("Strong data augmentation used in {}:".format('test' if test_flag else 'train'))
#             print(transform_list_strong)
#         else:
#             print("Data augmentation used in {}:".format('test' if test_flag else 'train'))
#             print(transform_list_strong)

#         if not test_flag:
#             self.transform_weak = self.get_transform(transform_list_weak)
#         self.transform_strong = self.get_transform(transform_list_strong)

#         self.test_flag = test_flag

#     @staticmethod
#     def get_transform(transform_list):
        
#         tfs = []
#         for key, value in zip(transform_list.keys(), transform_list.values()):
#             if value is not None:
#                 tf = eval(key)(**value)
#             else:
#                 tf = eval(key)()
#             tfs.append(tf)
#         return transforms.Compose(tfs)

#     def __getitem__(self, index):
#         image = Image.open(self.images[index]).convert('RGB')
#         gt = Image.open(self.gts[index]).convert('L')
#         shape = gt.size[::-1]
#         name = self.images[index].split('/')[-1]
#         if name.endswith('.jpg'):
#             name = name.split('.jpg')[0] + '.png'
            
#         sample_weak = {'image': image, 'gt': gt, 'name': name, 'shape': shape, 'index': index}
#         sample_strong = {'image': image, 'gt': gt, 'name': name, 'shape': shape, 'index': index}

#         # sample = self.transform(sample)
#         # if self.split == "train":
#         if not self.test_flag:
#             sample_weak = self.transform_weak(sample_weak)
#             sample_strong = self.transform_strong(sample_strong)
#             sample = {'image_strong': sample_strong['image'], 'image_weak': sample_weak['image'], 'gt': sample_weak['gt'],
#                     'name': name, 'shape': shape, 'index': index}
#         else:
#             sample_strong = self.transform_strong(sample_strong)
#             sample = {'image': sample_strong['image'], 'gt': sample_strong['gt'],
#                     'name': name, 'shape': shape, 'index': index}

#         #     if None not in (self.ops_weak, self.ops_strong):
#         #         sample = self.transform(sample, self.ops_weak, self.ops_strong)
#         #     else:
#         #         sample = self.transform(sample)
                
#         # get image stats
#         # print("Image stats: ", image.shape)
#         # print("R channel", torch.mean(image[0]), torch.std(image[0]))
#         # print("G channel", torch.mean(image[1]), torch.std(image[1]))
#         # print("B channel", torch.mean(image[2]), torch.std(image[2]))
        
#         return sample # image, gt, name, shape, index

#     def filter_files(self):
#         assert len(self.images) == len(self.gts)
#         images, gts = [], []
#         for img_path, gt_path in zip(self.images, self.gts):
#             img, gt = Image.open(img_path), Image.open(gt_path)
#             if img.size == gt.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#         self.images, self.gts = images, gts

#     def __len__(self):
#         return self.size

class GeneralSegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=None,
        # transform_list_train=None,
        # transform_list_test=None,
        label_flag=False,
        test_flag=False):

        # self.ops_weak = ops_weak
        # self.ops_strong = ops_strong

        root = os.path.join(args.root_path, 'test' if test_flag else 'train')
        image_root, gt_root = os.path.join(root, 'images'), os.path.join(root, 'masks')

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
        self.images = sorted(self.images)

        
        if not test_flag:
            # random select labeled images from self.images with args.labeled_ratio
            # args.labeled_ratio
            # self.labeled_names = random.sample(self.images, int(len(self.images) * args.labeled_ratio))
            # use the start of the list as labeled images, to avoid the randomness
            self.labeled_names = self.images[:int(len(self.images) * args.labeled_num)]

            # labeled_name_txt = os.path.join(root, 'labeled.txt')
            # if os.path.exists(labeled_name_txt):
            #     with open(labeled_name_txt, 'r') as f:
            #         self.labeled_names = f.read().splitlines()
            # else:
            #     raise ValueError("labeled.txt does not exist in {}".format(root))
            # # print(self.labeled_names)
            # self.labeled_names = [os.path.join(image_root, f) for f in self.labeled_names]
            self.unlabeled_images = [f for f in self.images if f not in self.labeled_names]

            self.images = self.labeled_names if label_flag else self.unlabeled_images
            # if label_flag:
            #     print("Labeled images: {}".format(len(self.images)))
            # else:
            #     print("Unlabeled images: {}".format(len(self.images)))
        
        # self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [os.path.join(gt_root, f.split('/')[-1]) for f in self.images]
        # make file names from k.png to k_bin_mask.png
        if 'MonuSeg' in args.root_path:
            self.gts = [f.split('.png')[0] + '_bin_mask.png' for f in self.gts]
        self.gts = sorted(self.gts)
        
        self.filter_files()
        
        self.size = len(self.images)

        transform_list = transform_list_train if not test_flag else transform_list_test
        transform_list['resize']['size'] = args.patch_size

        if not test_flag:
            logging.info("--------------- Train ------------------")
            # if not args.dilation_erosion:
            transform_list.pop('random_dilation_erosion', None)
            # if not args.img_enhance:
                # transform_list.pop('random_image_enhance', None)
            # if not args.scale_crop:
            transform_list.pop('random_scale_crop', None)
            # if not args.gaussian_blur:
            #     transform_list.pop('random_gaussian_blur', None)
            logging.info("Data augmentation used in {}:".format('test' if test_flag else 'train'))
            logging.info(transform_list)
        else:
            logging.info("--------------- Test ------------------")
            logging.info("Data augmentation used in {}:".format('test' if test_flag else 'train'))
            logging.info(transform_list)

        # if not test_flag:
        #     self.transform_weak = self.get_transform(transform_list_weak)
        # self.transform_strong = self.get_transform(transform_list_strong)
        self.transform = self.get_transform(transform_list)

        self.test_flag = test_flag

    @staticmethod
    def get_transform(transform_list):
        
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
            
        sample = {'image': image, 'gt': gt, 'image_weak': image, 'image_strong': image, 
                  'name': name, 'shape': shape, 'index': index}

        # sample = self.transform(sample)
        # if self.split == "train":
        if not self.test_flag:
            sample = self.transform(sample)
        else:
            sample = self.transform(sample)

        #     if None not in (self.ops_weak, self.ops_strong):
        #         sample = self.transform(sample, self.ops_weak, self.ops_strong)
        #     else:
        #         sample = self.transform(sample)
                
        # get image stats
        # print("Image stats: ", image.shape)
        # print("R channel", torch.mean(image[0]), torch.std(image[0]))
        # print("G channel", torch.mean(image[1]), torch.std(image[1]))
        # print("B channel", torch.mean(image[2]), torch.std(image[2]))
        
        return sample # image, gt, name, shape, index

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size

class GeneralSegDatasetUnlabeled(torch.utils.data.Dataset):
    def __init__(self, args, test_flag=False):
        root = os.path.join(args.data_dir, 'test' if test_flag else 'train_unlabeled')
        image_root = os.path.join(root, 'images')

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
        self.images = sorted(self.images)
        
        # self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.bmp')]
        # self.gts = sorted(self.gts)
        
        # self.filter_files()
        
        self.size = len(self.images)
        transform_list = transform_list_train if not test_flag else transform_list_test
        transform_list['resize']['size'] = [args.image_size, args.image_size]
        if not args.scale_crop:
            transform_list.pop('random_scale_crop', None)
        if not args.flip:
            transform_list.pop('random_flip', None)
        if not args.rotate:
            transform_list.pop('random_rotate', None)
        if not args.img_enhance:
            transform_list.pop('random_image_enhance', None)
        if not args.dilation_erosion:
            transform_list.pop('random_dilation_erosion', None)
            
        self.transform = self.get_transform(transform_list)

        print("Data augmentation used in {}:".format('test' if test_flag else 'train'))
        print(transform_list)

        self.test_flag = test_flag

    @staticmethod
    def get_transform(transform_list):
        
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        shape = image.size[::-1]
        # gt = Image.open(self.gts[index]).convert('L')
        # shape = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
            
        sample = {'image': image, 'name': name, 'shape': shape}

        sample = self.transform(sample)
        image = sample['image']
        # gt = sample['gt']
        # get image stats
        # print("Image stats: ", image.shape)
        # print("R channel", torch.mean(image[0]), torch.std(image[0]))
        # print("G channel", torch.mean(image[1]), torch.std(image[1]))
        # print("B channel", torch.mean(image[2]), torch.std(image[2]))
        
        if self.test_flag:
            # return (image, gt, name, shape)
            raise NotImplementedError
        return (image, None)

    # def filter_files(self):
    #     assert len(self.images) == len(self.gts)
    #     images, gts = [], []
    #     for img_path, gt_path in zip(self.images, self.gts):
    #         img, gt = Image.open(img_path), Image.open(gt_path)
    #         if img.size == gt.size:
    #             images.append(img_path)
    #             gts.append(gt_path)
    #     self.images, self.gts = images, gts

    def __len__(self):
        return self.size

from torch.utils.data.sampler import Sampler
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        print("Labeled images in batch: ", self.primary_batch_size)
        print("Unlabeled images in batch: ", self.secondary_batch_size)


        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

import itertools
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))


def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def debug_tile(out, size=(100, 100)):
    debugs = []
    for debs in out['debug']:
        debug = []
        for deb in debs:
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = (log - log.min()) / (log.max() - log.min())
            log *= 255
            log = log.astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)


class resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        if 'gt' in sample.keys():
            sample['gt'] = sample['gt'].resize(self.size, Image.BILINEAR)
        if 'mask' in sample.keys():
            sample['mask'] = sample['mask'].resize(self.size, Image.BILINEAR)

        return sample

class random_scale_crop:
    def __init__(self, range=[0.75, 1.25]):
        self.range = range

    def __call__(self, sample):
        scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt', 'contour']:
                    base_size = sample[key].size

                    scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                    sample[key] = sample[key].resize(scale_size)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_flip:
    def __init__(self, lr=True, ud=True):
        self.lr = lr
        self.ud = ud

    def __call__(self, sample):
        lr = np.random.random() < 0.5 and self.lr is True
        ud = np.random.random() < 0.5 and self.ud is True
        # lr = self.lr
        # ud = self.ud

        for key in sample.keys():
            if key in ['image', 'image_weak', 'image_strong', 'gt', 'contour']:
                sample[key] = np.array(sample[key])
                if lr:
                    sample[key] = np.fliplr(sample[key])
                if ud:
                    sample[key] = np.flipud(sample[key])
                sample[key] = Image.fromarray(sample[key])

        return sample

class random_rotate:
    def __init__(self, range=[0, 360], interval=1):
        self.range = range
        self.interval = interval

    def __call__(self, sample):
        rot = (np.random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'image_weak', 'image_strong', 'gt', 'contour']:
                    base_size = sample[key].size

                    sample[key] = sample[key].rotate(rot, expand=True)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, sample):
        image = sample['image_strong']
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)
        sample['image_strong'] = image

        return sample

class random_dilation_erosion:
    def __init__(self, kernel_range):
        self.kernel_range = kernel_range

    def __call__(self, sample):
        gt = sample['gt']
        gt = np.array(gt)
        key = np.random.random()
        # kernel = np.ones(tuple([np.random.randint(*self.kernel_range)]) * 2, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(*self.kernel_range), ) * 2)
        if key < 1/3:
            gt = cv2.dilate(gt, kernel)
        elif 1/3 <= key < 2/3:
            gt = cv2.erode(gt, kernel)

        sample['gt'] = Image.fromarray(gt)

        return sample

class random_gaussian_blur:
    def __init__(self, apply=False):
        self.apply = apply

    def __call__(self, sample):
        if self.apply:
            image = sample['image_strong']
            if np.random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
            sample['image_strong'] = image

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image_weak, image_strong = sample['image_weak'], sample['image_strong']

        sample['image'] = np.array(image, dtype=np.float32)
        sample['gt'] = np.array(gt, dtype=np.float32)
        sample['image_weak'] = np.array(image_weak, dtype=np.float32)
        sample['image_strong'] = np.array(image_strong, dtype=np.float32)
        
        return sample

class normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image_weak, image_strong = sample['image_weak'], sample['image_strong']
        image /= 255
        image_weak /= 255
        image_strong /= 255
        if self.mean is not None and self.std is not None:
            image -= self.mean
            image /= self.std

            image_weak -= self.mean
            image_weak /= self.std

            image_strong -= self.mean
            image_strong /= self.std

        # norm to [0, 1] if max value is 255
        if np.max(gt) == 255:
            gt /= 255
        
        sample['image'] = image
        sample['gt'] = gt
        sample['image_weak'] = image_weak
        sample['image_strong'] = image_strong

        return sample

class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image_weak, image_strong = sample['image_weak'], sample['image_strong']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image_weak = image_weak.transpose((2, 0, 1))
        image_weak = torch.from_numpy(image_weak).float()
        image_strong = image_strong.transpose((2, 0, 1))
        image_strong = torch.from_numpy(image_strong).float()

        
        gt = torch.from_numpy(gt)
        gt = gt.unsqueeze(dim=0)

        sample['image'] = image
        sample['gt'] = gt
        sample['image_weak'] = image_weak
        sample['image_strong'] = image_strong

        return sample 
# import numpy as np
# from PIL import Image
# import cv2
# import torch
# import torch.nn.functional as F
# from PIL import Image, ImageOps, ImageFilter, ImageEnhance


# def load_config(config_dir):
#     return ed(yaml.load(open(config_dir), yaml.FullLoader))


# def to_cuda(sample):
#     for key in sample.keys():
#         if type(sample[key]) == torch.Tensor:
#             sample[key] = sample[key].cuda()
#     return sample


# def clip_gradient(optimizer, grad_clip):
#     for group in optimizer.param_groups:
#         for param in group['params']:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)

# def debug_tile(out, size=(100, 100)):
#     debugs = []
#     for debs in out['debug']:
#         debug = []
#         for deb in debs:
#             log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
#             log = (log - log.min()) / (log.max() - log.min())
#             log *= 255
#             log = log.astype(np.uint8)
#             log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
#             log = cv2.resize(log, size)
#             debug.append(log)
#         debugs.append(np.vstack(debug))
#     return np.hstack(debugs)


# class resize:
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, sample):
#         if 'image' in sample.keys():
#             # print("Resizing image to: ", self.size)
#             sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
#         if 'gt' in sample.keys():
#             sample['gt'] = sample['gt'].resize(self.size, Image.BILINEAR)
#         if 'mask' in sample.keys():
#             sample['mask'] = sample['mask'].resize(self.size, Image.BILINEAR)

#         return sample

# class random_scale_crop:
#     def __init__(self, range=[0.75, 1.25]):
#         self.range = range

#     def __call__(self, sample):
#         scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
#         if np.random.random() < 0.5:
#             for key in sample.keys():
#                 if key in ['image', 'gt', 'contour']:
#                     base_size = sample[key].size

#                     scale_size = tuple((np.array(base_size) * scale).round().astype(int))
#                     sample[key] = sample[key].resize(scale_size)

#                     sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
#                                                     (sample[key].size[1] - base_size[1]) // 2,
#                                                     (sample[key].size[0] + base_size[0]) // 2,
#                                                     (sample[key].size[1] + base_size[1]) // 2))

#         return sample

# class random_flip:
#     def __init__(self, lr=True, ud=True):
#         self.lr = lr
#         self.ud = ud

#     def __call__(self, sample):
#         lr = np.random.random() < 0.5 and self.lr is True
#         ud = np.random.random() < 0.5 and self.ud is True

#         for key in sample.keys():
#             if key in ['image', 'gt', 'contour']:
#                 sample[key] = np.array(sample[key])
#                 if lr:
#                     sample[key] = np.fliplr(sample[key])
#                 if ud:
#                     sample[key] = np.flipud(sample[key])
#                 sample[key] = Image.fromarray(sample[key])

#         return sample

# class random_rotate:
#     def __init__(self, range=[0, 360], interval=1):
#         self.range = range
#         self.interval = interval

#     def __call__(self, sample):
#         rot = (np.random.randint(*self.range) // self.interval) * self.interval
#         rot = rot + 360 if rot < 0 else rot

#         if np.random.random() < 0.5:
#             for key in sample.keys():
#                 if key in ['image', 'gt', 'contour']:
#                     base_size = sample[key].size

#                     sample[key] = sample[key].rotate(rot, expand=True)

#                     sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
#                                                     (sample[key].size[1] - base_size[1]) // 2,
#                                                     (sample[key].size[0] + base_size[0]) // 2,
#                                                     (sample[key].size[1] + base_size[1]) // 2))

#         return sample

# class random_image_enhance:
#     def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
#         self.enhance_method = []
#         if 'contrast' in methods:
#             self.enhance_method.append(ImageEnhance.Contrast)
#         if 'brightness' in methods:
#             self.enhance_method.append(ImageEnhance.Brightness)
#         if 'sharpness' in methods:
#             self.enhance_method.append(ImageEnhance.Sharpness)

#     def __call__(self, sample):
#         image = sample['image']
#         np.random.shuffle(self.enhance_method)

#         for method in self.enhance_method:
#             if np.random.random() > 0.5:
#                 enhancer = method(image)
#                 factor = float(1 + np.random.random() / 10)
#                 image = enhancer.enhance(factor)
#         sample['image'] = image

#         return sample

# class random_dilation_erosion:
#     def __init__(self, kernel_range):
#         self.kernel_range = kernel_range

#     def __call__(self, sample):
#         gt = sample['gt']
#         gt = np.array(gt)
#         key = np.random.random()
#         # kernel = np.ones(tuple([np.random.randint(*self.kernel_range)]) * 2, dtype=np.uint8)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(*self.kernel_range), ) * 2)
#         if key < 1/3:
#             gt = cv2.dilate(gt, kernel)
#         elif 1/3 <= key < 2/3:
#             gt = cv2.erode(gt, kernel)

#         sample['gt'] = Image.fromarray(gt)

#         return sample

# class random_gaussian_blur:
#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         image = sample['image']
#         if np.random.random() < 0.5:
#             image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
#         sample['image'] = image

#         return sample

# class tonumpy:
#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         image, gt = sample['image'], sample['gt']

#         sample['image'] = np.array(image, dtype=np.float32)
#         sample['gt'] = np.array(gt, dtype=np.float32)
        
#         return sample

# class normalize:
#     def __init__(self, mean=None, std=None):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample):
#         image, gt = sample['image'], sample['gt']
#         image /= 255
#         if self.mean is not None and self.std is not None:
#             image -= self.mean
#             image /= self.std

#         # norm to [0, 1] if max value is 255
#         if np.max(gt) == 255:
#             gt /= 255
        
#         sample['image'] = image
#         sample['gt'] = gt

#         return sample

# class totensor:
#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         image, gt = sample['image'], sample['gt']

#         image = image.transpose((2, 0, 1))
#         image = torch.from_numpy(image).float()
        
#         gt = torch.from_numpy(gt)
#         gt = gt.unsqueeze(dim=0)

#         sample['image'] = image
#         sample['gt'] = gt

#         return sample