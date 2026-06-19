from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from lightning import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from ev_at.core.logging import get_logger
from ev_at.data.transforms import AugMix, AutoAugCifar10, Cutout

from .dataset import CIFARCorruptedDatasetWrapper, CIFARDatasetWrapper

logger = get_logger("data_loader")

__all__ = ["CIFAR10Datamodule", "CIFAR10CorruptedDatamodule"]


class CIFARDatamodule(ABC, LightningDataModule):
    """
    Abstract data module for CIFAR dataset.

    This class handles the setup of training, validation, and testing datasets,
    and provides data loaders for each of these datasets.

    It enables better integration with PyTorch Lightning by encapsulating
    the dataset loading and preprocessing logic, making it easier to manage
    the data pipeline for training and evaluation.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 8,
        aug: Literal["basic", "autoaug", "cutout", "augmix"] = "basic",
    ):
        """
        Args:
            data_path (str | Path): Path to the directory containing the dataset.
            batch_size (int): Batch size for data loaders.
            seed (int): Random seed for reproducibility.
            num_workers (int): Number of worker processes for data loading.
            aug (Literal["basic", "autoaug", "cutout", "augmix"]): Data augmentation strategy.
        """
        super().__init__()
        self._data_path = data_path
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers
        if aug == "basic":
            self._transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif aug == "cutout":
            self._transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                ]
            )
        elif aug == "autoaug":
            self._transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    AutoAugCifar10(),
                    transforms.ToTensor(),
                ]
            )
        elif aug == "augmix":
            self._transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    AugMix(),
                ]
            )

        self._transform_test = transforms.Compose([transforms.ToTensor()])

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: Literal["fit", "val", "test"]):
        pass

    def train_dataloader(self):
        torch.manual_seed(self._seed)
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # No validation set is used with CIFAR10.
        # We use the test set for validation during training.
        # As in the literature.
        return DataLoader(
            self.dataset_test,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )


class CIFAR10Datamodule(CIFARDatamodule):
    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 8,
        aug: Literal["basic", "autoaug", "cutout", "augmix"] = "basic",
    ):
        super().__init__(data_path, batch_size, seed, num_workers, aug)

    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download the dataset if it doesn't exist already."""
        CIFAR10(
            root=self._data_path,
            download=True,
        )
        CIFAR10(
            root=self._data_path,
            train=False,
            download=True,
        )

    def setup(self, stage: Literal["fit", "val", "test"]):
        """Setup the datasets based on the stage.

        Args:
            stage (Literal["fit", "val", "test"]): The stage for which to set up the datasets.
                'fit' for training, 'val' for validation, 'test' for testing.
        """
        match stage:
            case "fit":
                self.dataset_train = CIFARDatasetWrapper(
                    CIFAR10(
                        root=self._data_path,
                        train=True,
                        transform=self._transform_train,
                        download=False,
                    )
                )
                self.dataset_test = CIFARDatasetWrapper(
                    CIFAR10(
                        root=self._data_path,
                        train=False,
                        transform=self._transform_test,
                        download=False,
                    )
                )
            case "test":
                self.dataset_test = CIFARDatasetWrapper(
                    CIFAR10(
                        root=self._data_path,
                        train=False,
                        transform=self._transform_test,
                        download=False,
                    )
                )


class CIFAR100Datamodule(CIFARDatamodule):
    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 8,
        aug: Literal["basic", "autoaug", "cutout", "augmix"] = "basic",
    ):
        super().__init__(data_path, batch_size, seed, num_workers, aug)

    def num_classes(self) -> int:
        return 100

    def prepare_data(self):
        """Download the dataset if it doesn't exist already."""
        CIFAR100(
            root=self._data_path,
            download=True,
        )
        CIFAR100(
            root=self._data_path,
            train=False,
            download=True,
        )

    def setup(self, stage: Literal["fit", "val", "test"]):
        """Setup the datasets based on the stage.

        Args:
            stage (Literal["fit", "val", "test"]): The stage for which to set up the datasets.
                'fit' for training, 'val' for validation, 'test' for testing.
        """
        match stage:
            case "fit":
                self.dataset_train = CIFARDatasetWrapper(
                    CIFAR100(
                        root=self._data_path,
                        train=True,
                        transform=self._transform_train,
                        download=False,
                    )
                )
                self.dataset_test = CIFARDatasetWrapper(
                    CIFAR100(
                        root=self._data_path,
                        train=False,
                        transform=self._transform_test,
                        download=False,
                    )
                )
            case "test":
                self.dataset_test = CIFARDatasetWrapper(
                    CIFAR100(
                        root=self._data_path,
                        train=False,
                        transform=self._transform_test,
                        download=False,
                    )
                )


class CIFARCorruptedDatamodule(ABC, LightningDataModule):
    """
    Data module for CIFAR10-C dataset.

    This class handles the setup of testing datasets with various corruptions,
    and provides test data loader.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        """
        Args:
            data_path (str | Path): Path to the directory containing the dataset.
            batch_size (int): Batch size for data loaders.
        """
        super().__init__()
        self._data_path = data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._transform_test = transforms.Compose([transforms.ToTensor()])

    @abstractmethod
    def num_classes(self) -> int:
        pass

    @abstractmethod
    def setup(self, stage: Literal["test"]):
        pass

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )


class CIFAR10CorruptedDatamodule(CIFARCorruptedDatamodule):
    def __init__(
        self, data_path: str | Path, batch_size: int = 32, num_workers: int = 8
    ):
        super().__init__(data_path, batch_size, num_workers)

    def num_classes(self) -> int:
        return 10

    def setup(self, stage: Literal["test"]):
        """
        Setup the datasets for testing.

        Args:
            stage (Literal["test"]): The stage for which to set up the datasets.
        """
        if stage == "test":
            self.dataset_test = CIFARCorruptedDatasetWrapper(
                load_dataset(
                    "randall-lab/cifar10-c", cache_dir=self._data_path, split="test"
                ),
                transform=self._transform_test,
            )


class CIFAR100CorruptedDatamodule(CIFARCorruptedDatamodule):
    def __init__(
        self, data_path: str | Path, batch_size: int = 32, num_workers: int = 8
    ):
        super().__init__(data_path, batch_size, num_workers)

    def num_classes(self) -> int:
        return 100

    def setup(self, stage: Literal["test"]):
        """
        Setup the datasets for testing.

        Args:
            stage (Literal["test"]): The stage for which to set up the datasets.
        """
        if stage == "test":
            self.dataset_test = CIFARCorruptedDatasetWrapper(
                load_dataset(
                    "randall-lab/cifar100-c", cache_dir=self._data_path, split="test"
                ),
                transform=self._transform_test,
            )
