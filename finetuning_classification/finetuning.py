import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode as IM
from torch.utils.data import DataLoader, Dataset
import torchvision
import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join
import cv2
import os
from typing import Tuple, Dict, List, Union, Type, Iterator
CNT_CLASSES = ...
class MyCustomDataset(Dataset):
    def __init__(
        self,
        mode: str = 'train', # {"train", "val", "test"},
        fraction: float = 0.7,
        dirname: str = None,
        labels: Dict[str, np.ndarray] = None,
        transform = None,
    ):
        self._dirname = dirname
        images_dir = sorted(os.listdir(dirname))
        if mode == "train":
            np.random.seed(1)
            np.random.shuffle(images_dir)
            partition = int(fraction * len(images_dir))
            self._filenames = images_dir[:partition]
        if mode == "val":
            np.random.seed(1)
            np.random.shuffle(images_dir)
            partition = int(fraction * len(images_dir))
            self._filenames = images_dir[partition:]
        if mode == "test":
            self._filenames = images_dir

        self._labels = labels.copy() if labels is not None else {
            filename: np.zeros(1) for filename in self._filenames
        }
        self._transform = transform

    def __len__(self) -> int:
        return len(self._filenames)

    def __iter__(self) -> Iterator[Tuple[Tensor, np.ndarray, str]]:
        yield from zip(
            map(self._image_reader, self._filenames),
            map(self._labels_getter, self._filenames),
            self._filenames)

    def __getitem__(self, index) -> Tuple[Tensor, np.ndarray, str]:
        img_filename = self._filenames[index]
        return self._image_reader(img_filename), self._label_getter(img_filename), img_filename

    def _image_reader(self, img_filename: str) -> Tensor:
        filename = join(self._dirname, img_filename)
        image = Image.open(filename).convert('RGB')
        ## augmentation
        x = self._transform(image)
        return x

    def _label_getter(self, img_filename: str) -> np.ndarray:
        return self._labels[img_filename]

    @staticmethod
    def collate_fn(items: List[Tuple[Tensor, np.ndarray, str]]):
        images = []
        labels = []
        filenames = []

        for image, label, filename in items:
            images.append(image)
            labels.append(torch.from_numpy(np.array(label)))
            filenames.append(filename)

        images = torch.stack(images)
        labels = torch.stack(labels)

        return images, labels, filenames

class EfficientNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, count_non_freeze=0):
        super().__init__()

        self.efficientnet_b1 =  torchvision.models.swin_v2_t(weights='IMAGENET1K_V1')
        linear_size=768
        if len(list(self.efficientnet_b1.children())) < count_non_freeze:
            raise NotImplementedError("Wrong freezing parameter")
        for child in list(self.efficientnet_b1.children()):
            for param in child.parameters():
                param.requires_grad = True

        for child in list(self.efficientnet_b1.children())[:-count_non_freeze]:
            for param in child.parameters():
                param.requires_grad = False
        self.efficientnet_b1.head = nn.Sequential(
             nn.Linear(linear_size, num_classes),
        )
        
    def forward(self, x):
        return F.log_softmax(self.efficientnet_b1(x), dim=1)

class BirdsClassifier(pl.LightningModule):
    def __init__(self, lr_rate=1e-4, num_classes=CNT_CLASSES, count_non_freeze=6):
        super().__init__()
        self.model = EfficientNetClassifier(num_classes, count_non_freeze)
        self.lr_rate = lr_rate

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        self.train()
        x, y, file = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        acc = torch.sum(y_pred.detach().argmax(dim = 1) == y.detach()) / y.shape[0]
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=len(y))
        torch.cuda.empty_cache()
        return loss
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4, weight_decay = 5e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=5,
                                                                            T_mult=1,
                                                                            eta_min=4e-3,
                                                                            last_epoch=-1,
                                                                            verbose=True)
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss"
        }

        return [optimizer], [lr_dict]
    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        self.eval()

        x, y, file = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        acc = torch.sum(y_pred.detach().argmax(dim = 1) == y.detach()) / y.shape[0]
        torch.cuda.empty_cache()
        metrics =  {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=len(y))
        return metrics

from torchvision import transforms
def my_custom_normalize(img):
    for c in range(3):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        img[:, :, c] -= mean[c]
        img[:, :, c] /= std[c]
    return img
def train_classifier(labels: Dict[str, np.ndarray], train_data_dir: str):
    epochs = 100
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    num_workers = 2
    transform = transforms.Compose([
        transforms.Resize(size=256, interpolation=IM.BICUBIC),
        # Обрезаем "лишние" пиксели. Если их нет, то CenterCrop ничего не изменит (случай "меньше").
        transforms.CenterCrop(256),
        # Преобразуем PIL.Image изображение в массив np.array
        transforms.Lambda(lambda x: np.array(x).astype(np.float32) / 255.0),
        my_custom_normalize,
        transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1))),

    ])
    batch_size = 64

    # Init train and val datasets
    ds_train = MyCustomDataset(
        mode = "train",
        dirname = train_data_dir,
        labels=labels,
        transform = transform,
    )
    ds_val = MyCustomDataset(
        mode = "val",
        dirname = train_data_dir,
        labels=labels,
        transform = transform,
    )

    # Init train and val dataloaders
    dl_train = DataLoader(
        ds_train,
        collate_fn=MyCustomDataset.collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    dl_val = DataLoader(
        ds_val,
        collate_fn=MyCustomDataset.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    trainer = pl.Trainer(
        max_epochs = epochs,
        accelerator = accelerator,
        callbacks = None
    )
    model = BirdsClassifier(fast_train=fast_train, num_classes=50, count_non_freeze=4)
    trainer.fit(model, dl_train, dl_val)
    return model

