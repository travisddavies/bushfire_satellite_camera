import torch
from torchvision import transforms


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class DualToTensor:
    def __call__(self, img, mask):
        return transforms.ToTensor()(img), torch.as_tensor(mask, dtype=torch.uint8)


class DualRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.size, self.size))
        img_cropped = transforms.functional.crop(img, i, j, h, w)
        mask_cropped = transforms.functional.crop(mask, i, j, h, w)

        return img_cropped, mask_cropped


class DualRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, img, mask):
        img_flipped = transforms.RandomHorizontalFlip(self.prob)(img)
        mask_flipped = transforms.RandomHorizontalFlip(self.prob)(mask)

        return img_flipped, mask_flipped


class DualRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, img, mask):
        img_flipped = transforms.RandomVerticalFlip(self.prob)(img)
        mask_flipped = transforms.RandomVerticalFlip(self.prob)(mask)

        return img_flipped, mask_flipped


def get_dual_transform(size, p):
    transform = DualCompose(
        DualToTensor(),
        DualRandomCrop(size),
        DualRandomVerticalFlip(p),
        DualRandomHorizontalFlip(p)
    )

    return transform
