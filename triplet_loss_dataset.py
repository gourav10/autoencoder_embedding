from torchvision.datasets import ImageFolder
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json
from torchvision.transforms import transforms
import random
import torch.nn as nn


class TripletDataset(Dataset):
    def __init__(self, mnist_dataset, is_train=True, transform=None):
        self.mnist_data = mnist_dataset
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, index):
        anchor_img, anchor_label = self.mnist_data[index]
        # anchor_img = anchor_img.convert('RGB')
        positive_img, positive_label = self.get_positive_data(
            anchor_label, index
        )
        # positive_img = positive_img.convert('RGB')
        negative_img, negative_label = self.get_negative_data(
            anchor_label, index
        )
        # negative_img = negative_img.convert('RGB')
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img, anchor_label

    def get_positive_data(self, anchor_label, anchor_index):
        positive_list = []
        for idx, item in enumerate(self.mnist_data):
            image, label = item
            if idx == anchor_index:
                continue
            if label == anchor_label:
                positive_list.append(item)
        positive_item = random.choice(positive_list)
        return positive_item

    def get_negative_data(self, anchor_label, anchor_index):
        negative_list = []
        for idx, item in enumerate(self.mnist_data):
            image, label = item
            if idx == anchor_index:
                continue
            if label != anchor_label:
                negative_list.append(item)
        negative_item = random.choice(negative_list)
        return negative_item
