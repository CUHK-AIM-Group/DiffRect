import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from PIL import Image

from skimage import transform as sk_trans
# from scipy.ndimage import rotate, zoom

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "valtest":
            with open(self._base_dir + "/valtest.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

        self.mrseg19norm = MRSEG19Normalization()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:] # np array
        if np.max(image) > 1:
            image = self.mrseg19norm(image, mode='Max_Min')
        label = h5f["label"][:] # np array
        
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample["case"] = case
        return sample

class MRSEG19Normalization(object):
    def __call__(self, image, mode):

        if mode == 'Max_Min':
            eps = 1e-8
            mn = image.min()
            mx = image.max()
            image_normalized = (image - mn) / (mx - mn + eps)
        
        if mode == 'Zero_Mean_Unit_Std':
            eps = 1e-8
            mean = image.mean()
            std = image.std()
            image_normalized = (image - mean) / (std + eps)

        if mode == 'Truncate':
            # truncate
            
            Hist, _ = np.histogram(image, bins=int(image.max()))

            idexs = np.argwhere(Hist >= 20)
            idex_min = np.float32(0)
            idex_max = np.float32(idexs[-1, 0])

            image[np.where(image <= idex_min)] = idex_min
            image[np.where(image >= idex_max)] = idex_max

            # normalize
            sig = image[0, 0, 0]
            image = np.where(image != sig, image - np.mean(image[image != sig]), 0 * image)
            image_normalized = np.where(image != sig, image / np.std(image[image != sig] + 1e-8), 0 * image)
        
        return image_normalized
    
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        # print("image", image.shape, image.max(), image.min())
        # print("label", label.shape, label.max(), label.min())
        sample = {"image": image, "label": label}
        return sample

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
class c_random_flip:
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
    
class c_random_rotate:
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
    
class c_normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image_weak, image_strong = sample['image_weak'], sample['image_strong']
        if image.max() > 1:
            image /= 255
            image_weak /= 255
            image_strong /= 255
        else:
            # print('image max value is {:.3f}'.format(np.max(image)))
            pass
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
    
class c_random_image_enhance:
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
    
class c_random_gaussian_blur:
    def __init__(self, apply=False):
        self.apply = apply

    def __call__(self, sample):
        if self.apply:
            image = sample['image_strong']
            if np.random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
            sample['image_strong'] = image

        return sample
    
class c_tonumpy:
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
    
class WeakStrongAugment_Ours(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size, args=None, split='train'):
        self.output_size = output_size
        self.split = split

        self.transform_list_train = {
                            'c_random_flip': {'lr': True, 'ud': True}, 
                            'c_random_rotate': {'range': [0, 359]}, 
                            'c_random_image_enhance': {'methods': ['contrast', 'sharpness', 'brightness']}, 
                            'c_random_gaussian_blur': {'apply': True},
                            'c_tonumpy': None, 
                            'c_normalize': {'mean': None, 'std': None},}
        self.transform_list_test = {
                            'c_tonumpy': None, 
                            'c_normalize': {'mean': None, 'std': None},}

        if args:
            if args.no_color:
                self.transform_list_train.pop('c_random_image_enhance', None)
            if args.no_blur:
                self.transform_list_train.pop('c_random_gaussian_blur', None)
            if args.rot != 359:
                self.transform_list_train['c_random_rotate']['range'] = [-args.rot, args.rot]
            # if 

        self.transform_list_train = self.get_transform(self.transform_list_train)
        self.transform_list_test = self.get_transform(self.transform_list_test)

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

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        image_untrans_np = image
        image = torch.from_numpy(image.astype(np.float32))
        image = transforms.ToPILImage()(image)
        image_weak = torch.from_numpy(image_untrans_np.astype(np.float32))
        image_weak = transforms.ToPILImage()(image_weak)
        image_strong = torch.from_numpy(image_untrans_np.astype(np.float32))
        image_strong = transforms.ToPILImage()(image_strong)
        label = torch.from_numpy(label.astype(np.uint8))
        label = transforms.ToPILImage()(label)
        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "gt": label,
        }
        if self.split == 'train':
            transform_list = self.transform_list_train
        else:
            transform_list = self.transform_list_test
        
        sample_new = transform_list(sample)
        sample_new['image'] = torch.from_numpy(sample_new['image'].astype(np.float32)).unsqueeze(0)
        sample_new['image_weak'] = torch.from_numpy(sample_new['image_weak'].astype(np.float32)).unsqueeze(0)
        sample_new['image_strong'] = torch.from_numpy(sample_new['image_strong'].astype(np.float32)).unsqueeze(0)
        sample_new['label_aug'] = torch.from_numpy(sample_new['gt'].astype(np.uint8))

        return sample_new

    def resize(self, image):
        if len(image.shape) == 3:
            x, y, z = image.shape 
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        elif len(image.shape) == 2:
            x, y = image.shape 
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


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


class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

