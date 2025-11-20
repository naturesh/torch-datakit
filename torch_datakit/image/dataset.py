import glob
import random

from PIL import Image
from torch.utils.data import Dataset, Subset

__all__ = ["FolderDataset", "TransformWrapper", "RandomSampling"]


class FolderDataset(Dataset):
    def __init__(self, paths: str | list, labels: int | list):
        if isinstance(paths, str):
            self.paths = glob.glob(paths)
        else:
            self.paths = paths

        if isinstance(labels, int):
            self.labels = len(self.paths) * [labels]
        else:
            self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return Image.open(self.paths[idx]).convert("RGB"), self.labels[idx]


class TransformWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        return self.transform(img), label


def RandomSampling(dataset, n):
    return Subset(dataset, random.sample(range(len(dataset)), n))
