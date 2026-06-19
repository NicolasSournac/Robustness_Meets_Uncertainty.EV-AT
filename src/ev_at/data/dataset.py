from torch.utils.data import Dataset

class CIFARDatasetWrapper(Dataset):
    """A simple dataset wrapper for CIFAR10"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset.__getitem__(idx)
        return data, target, idx
    

class CIFARCorruptedDatasetWrapper(Dataset):
    """A dataset wrapper for CIFAR-C that applies a transform to the data."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        data = example["image"]
        target = example["label"]
        data = self.transform(data)
        return data, target, idx