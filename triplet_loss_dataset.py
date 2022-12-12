from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms


class Triplet_MNIST(Dataset):
    def __init__(self, df, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.images = df.iloc[:, 1:].values.astype(np.uint8)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values
        else:
            self.images = df.values.astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        anchor_img = self.images[index].reshape(28, 28, 1)

        if self.is_train:
            anchor_label = self.labels[index]

            positive_list = self.index[self.index != index][
                self.labels[self.index != index] == anchor_label
            ]
            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(28, 28, 1)

            negative_list = self.index[self.index != index][
                self.labels[self.index != index] != anchor_label
            ]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(28, 28, 1)

            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
                positive_img = self.transform(self.to_pil(positive_img))
                negative_img = self.transform(self.to_pil(negative_img))

            return anchor_img, positive_img, negative_img, anchor_label
        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img
