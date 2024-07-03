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