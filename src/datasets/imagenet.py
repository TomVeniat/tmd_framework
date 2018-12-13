from torchvision.datasets import ImageFolder
from pathlib import Path
import torchvision

def init_imagenet(root: str, tiny: bool = True, im_size: int = 64, train=True):

    if train:
        root = Path(root)/Path('train')
    else:
        root = Path(root)/Path('test')


    normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if tiny:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    else:
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(im_size*1.14),
            torchvision.transforms.RandomResizedCrop(im_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return ImageFolder(root, transforms)

