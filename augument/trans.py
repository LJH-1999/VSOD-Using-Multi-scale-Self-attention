import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms as tfs

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(image, self.size)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask

class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue):
        self.cj = tfs.ColorJitter(brightness=brightness,
                                  contrast=contrast,
                                  saturation=saturation,
                                  hue=hue)

    def __call__(self, image, mask):
        image = self.cj(image)
        return image, mask

class RandomErasing:
    def __init__(self, p, scale, ratio):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, mask):
        if image.mode != 'RBG':
            return image, mask
        if random.random() < self.p:
            params = tfs.RandomErasing.get_params(image,
                                                  scale=self.scale,
                                                  ratio=self.ratio,
                                                  )
            image = F.erase(image, *params)
            mask = F.erase(mask, *params)
        return image, mask

class RandomResizedCrop:
    def __init__(self, size, scale, ratio):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self, image, mask):
        params = tfs.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = F.resized_crop(image, *params, size=[self.size, self.size])
        mask = F.resized_crop(mask, *params, size=[self.size, self.size])
        return image, mask
class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):
        degrees = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, degrees)
        mask = F.rotate(mask, degrees)
        return image, mask


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target


class Normalize:
    def __init__(self):
        self.mean1 = [0.485, 0.456, 0.406]
        self.std1 = [0.229, 0.224, 0.225]
        self.mean2 = [0.449]
        self.std2=[0.226]

    def __call__(self, image, target):
        if image.shape[0] == 3:
            image = F.normalize(image, mean=self.mean1, std=self.std1)
        else:
            image = F.normalize(image, mean=self.mean2, std=self.std2)
            image = torch.concat((image, image, image))
        return image, target


class Pad:
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
