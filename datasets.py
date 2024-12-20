import os
import random
import cv2
from PIL import Image
import numpy as np
import torch.utils.data as data
import albumentations as A
from torchvision.transforms import ToTensor, Normalize, Compose
from tqdm import tqdm

combo_train = A.Compose([
    A.OneOf([
        A.RandomCrop(224, 224, p=0.8),
        A.Resize(224, 224, p=0.2)
    ], p=1),
    A.RandomBrightnessContrast(),
    A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=30, val_shift_limit=0),
])

combo_val = A.Compose([
    A.Resize(224, 224)
])


class BaseDateset(data.Dataset):
    def __init__(self, config, mode='train', transform=None):
        """
        :param root:
        :param list_txt:
        :param classes: [name1, name2, ....]
        :param transform:
        """
        self.root = config['data_root']
        self.images = []
        self.labels = []

        if mode == 'train':
            list_file = config['train_txt']
        else:
            list_file = config['val_txt']
        self.parse(list_file)

        self.transform = transform 
        self.normal = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  # RGB ImageNet Pretrain model

    def parse(self, txt):
        with open(txt, 'r') as f:
            lines = f.readlines()
            f.close()
        for line in tqdm(lines):
            line = line.strip()
            rel_path = line.split(' ')[0]
            image_path = os.path.join(self.root, rel_path)
            self.images.append(image_path)
            self.labels.append(int(line.split(' ')[-1]))

    def __getitem__(self, i):
        img = np.array(Image.open(self.images[i]).convert("RGB"))
        label = self.labels[i]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return self.normal(img), label

    def __len__(self):
        return len(self.images)


class RegDataset(BaseDateset):
    def parse(self, txt):
        with open(txt, 'r') as f:
            lines = f.readlines()
            f.close()
        for line in tqdm(lines):
            line = line.strip()
            rel_path = line.split(' ')[0]
            image_path = os.path.join(self.root, rel_path)
            self.images.append(image_path)
            self.labels.append(np.float32(line.split(' ')[-1]))


class RegDatasetWithPath(RegDataset):
    def __getitem__(self, i):
        img = np.array(Image.open(self.images[i]).convert("RGB"))
        label = self.labels[i]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return self.normal(img), label, self.images[i]


if __name__ == "__main__":
    pass
