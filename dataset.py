# dataset.py 
import os 
import numpy as np 
import torch
import cv2 
import matplotlib.pyplot as plt

from torch.utils import data 
from PIL import Image 
import albumentations as A
import torchvision.transforms as transforms
import random


# +
class Flickr(data.Dataset): 
    def __init__(self, path, image_set, transforms=None, mixup = False, alpha = 0.2): 
        super(Flickr, self).__init__() 
        assert image_set in ('train', 'val'), "image_set is not valid!"
        self.data_path = path 
        self.image_set = image_set
        self.transforms = transforms 
        self.mixup = mixup
        self.alpha = alpha
        self.classes = ['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper',
           'plastic', 'stone', 'water', 'wood']
        self.map = {'fabric': 0, 'foliage': 1, 'glass': 2, 'leather': 3, 
                    'metal': 4, 'paper': 5, 'plastic': 6, 'stone': 7,
                   'water': 8, 'wood': 9}
        self.createIndex()

    def createIndex(self): 
        self.img_list = [] 
        self.label_list = [] 
        if self.image_set == 'train': 
            for c in self.classes: 
                for root, dirs, files in os.walk(os.path.join(
                    os.getcwd(), 'FMD', 'image','train', '{}'.format(c))):
                    for f in files:
                        if f.endswith('.jpg'): 
                            s = os.path.join(root, f)
                            self.img_list.append(s)
                            self.label_list.append(self.map[c])
        if self.image_set == 'val': 
            for c in self.classes: 
                for root, dirs, files in os.walk(os.path.join(
                    os.getcwd(), 'FMD', 'image','test', '{}'.format(c))):

                    for f in files:
                        if f.endswith('.jpg'):
                            s = os.path.join(root, f)
                            self.img_list.append(s)
                            self.label_list.append(self.map[c])
    
    def mixing_up(self, img, idx):
        img2_idx = random.randint(0, len(self.img_list)-1)
        img2 = self.img_list[img2_idx]
        img2 = cv2.imread(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2_label = self.label_list[img2_idx]
        mixed_img = img*(1-self.alpha) + img2*self.alpha
        mixed_img = np.round_(mixed_img)
        mixed_img = mixed_img.astype(np.uint8)
        mixed_label = self.label_list[idx] if self.alpha < 0.5 else self.label_list[img2_idx]
        return mixed_img, mixed_label 

    def __getitem__(self, idx): 
        img = self.img_list[idx]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[idx]
        if self.mixup and random.random() > 0.5: # comment out (and random.random() > 0.5) to remove the 50% of mixing up
            img, label = self.mixing_up(img, idx)
        img = Image.fromarray(img)        
        plt.imshow(img)
        assert label in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "Label is not valid, please check"
        if self.transforms is not None: 
            if type(self.transforms) == A.Compose:
                img = transforms.functional.resize(img, (227, 227))
                img = self.transforms(image = np.array(img))['image']
                img = transforms.functional.to_tensor(img)
            else:
                img = self.transforms(img) 
        sample = {
            'img': img, 
            'label': label,
        }
        return sample 

    def __len__(self): 
        return len(self.img_list)         


class MINC(data.Dataset): 
    def __init__(self, path, image_set, transforms=None): 
        super(MINC, self).__init__() 
        assert image_set in ('train', 'val'), "image_set is not valid!"
        self.data_path = path 
        self.image_set = image_set
        self.transforms = transforms 
        self.classes = ['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper',
           'plastic', 'stone', 'water', 'wood']
        self.map = {'fabric': 0, 'foliage': 1, 'glass': 2, 'leather': 3, 
                    'metal': 4, 'paper': 5, 'plastic': 6, 'stone': 7,
                   'water': 8, 'wood': 9}
        self.createIndex() 
    def createIndex(self): 
        self.img_list = [] 
        self.label_list = [] 
        if self.image_set == 'train': 
            with open(os.path.join(os.getcwd(), 'minc-2500', 'labels', 'train.txt'), 'r') as f:
                for line in f: 
                    line = line.split(" ")
                    self.img_list.append(os.path.join(os.getcwd(), 'minc-2500', line[0]))
                    self.label_list.append(int(line[1]))
        if self.image_set == 'val': 
            with open(os.path.join(os.getcwd(), 'minc-2500', 'labels', 'test.txt'), 'r') as f:
                for line in f: 
                    line = line.split(" ")
                    self.img_list.append(os.path.join(os.getcwd(), 'minc-2500', line[0]))
                    self.label_list.append(int(line[1]))

    def __getitem__(self, idx): 
        img = self.img_list[idx]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        label = self.label_list[idx]
        plt.imshow(img)
        assert label in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "Label is not valid, please check"
        if self.transforms is not None: 
            if type(self.transforms) == A.Compose:
                img = transforms.functional.resize(img, (227, 227))
                img = self.transforms(image = np.array(img))['image']
                img = transforms.functional.to_tensor(img)
            else:
                img = self.transforms(img) 
        sample = {
            'img': img, 
            'label': label,
        }
        return sample 

    def __len__(self): 
        return len(self.img_list)         
