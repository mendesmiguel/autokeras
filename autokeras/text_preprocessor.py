import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose

from autokeras.constant import Constant


class DataTransformer:
    def __init__(self, data):

        self.max_val = data.max()

    def transform_train(self, data, targets=None, batch_size=None):
        if not self.augment:
            augment_list = []
        else:
            augment_list = [ToPILImage(),
                            RandomCrop(data.shape[1:3], padding=4),
                            RandomHorizontalFlip(),
                            ToTensor()
                            ]
        common_list = [Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))]
        compose_list = augment_list + common_list

        return self._transform(batch_size, compose_list, data, targets)

    def transform_test(self, data, targets=None, batch_size=None):
        common_list = [Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))]
        compose_list = common_list

        return self._transform(batch_size, compose_list, data, targets)

    def _transform(self, batch_size, compose_list, data, targets):
        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)
        data = data / self.max_val
        data = torch.Tensor(data.transpose(0, 3, 1, 2))
        data_transforms = Compose(compose_list)
        dataset = MultiTransformDataset(data, targets, data_transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MultiTransformDataset(Dataset):
    def __init__(self, dataset, target, compose):
        self.dataset = dataset
        self.target = target
        self.compose = compose

    def __getitem__(self, index):
        feature = self.dataset[index]
        if self.target is None:
            return self.compose(feature)
        return self.compose(feature), self.target[index]

    def __len__(self):
        return len(self.dataset)
