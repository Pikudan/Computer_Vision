import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import albumentations as A
from skimage import io
import numpy as np
from PIL import Image
import cv2
import os
from os.path import abspath, dirname, join
from pytorch_lightning.callbacks import ModelCheckpoint
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

INPUT_SIZE = 284

def normalize(img):
    for c in range(3):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        img[:, :, c] -= mean[c]
        img[:, :, c] /= std[c]
    return img

def load_resize(img_path, label, inference = False):
    image = io.imread(img_path)
    image = np.array(image).astype(np.float32)
    img_shape = image.shape
    if len(img_shape) != 3:
       image = np.stack((image,)*3, axis=-1)
    # resize
    if not inference:
        target = np.copy(label)
        target[::2] = label[::2] / image.shape[1] * INPUT_SIZE
        target[1::2] = label[1::2] / image.shape[0] * INPUT_SIZE

    x = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

    if not inference:
        return x, target
    else:
        return x, image.shape
class MyCustomDataset(Dataset):
    def __init__(self,
                 mode,
                 fraction: float = 0.9,
                 transform = None,
                 store_in_ram = True,
                 data_dir = None,
                 train_gt = None,
                 inference = False,
                ):

        self._items = []
        self._transform = transform
        self.store_in_ram = store_in_ram
        self.inference = inference
        images_dir = os.listdir(data_dir)
        if not inference:
            np.random.seed(1)
            np.random.shuffle(images_dir)

        labels = train_gt

        partition = int(fraction * len(images_dir))
        if mode == 'train':
            img_names = images_dir[:partition]
        elif mode == 'val':
            img_names = images_dir[partition:]

        for img_name in img_names:
            if not inference:
                label = np.array(labels[img_name]).astype(np.float32)
            else:
                label = 0

            img_path = os.path.join(data_dir, img_name)

            if self.store_in_ram and not inference:
                x, target = load_resize(img_path, label)

                if np.max(target) > INPUT_SIZE or np.min(target) < 0:
                    continue
            else:
                x = img_path
                target = label

            self._items.append((
                x,
                target
            ))
    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img, target = self._items[index]
        if not self.store_in_ram or self.inference:
            img, target = load_resize(img, target, inference = self.inference)

        if self._transform and not self.inference:
            keypoints = np.zeros((len(target) // 2, 2))
            keypoints[:, 0] = target[::2]
            keypoints[:, 1] = target[1::2]
            transformed = self._transform(image = img, keypoints = keypoints)
            if len(transformed['keypoints']) != 14:
                img = np.copy(img).astype(np.float32)
            else:
                img = transformed['image'].astype(np.float32)
                target = np.array(transformed['keypoints'], dtype = np.float32).ravel()

        img = normalize(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, target

import torch.nn as nn
class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        """ Define computations here. """

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = torch.nn.ReLU()
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d((2, 2), (2, 2))

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = torch.nn.ReLU()
        self.batch_norm3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d((2, 2), (2, 2))

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.relu4 = torch.nn.ReLU()
        self.batch_norm4 = torch.nn.BatchNorm2d(256)

        self.pool4 = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = torch.nn.Linear(256 * 7 * 7, 64)
        self.relu5 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.2),
        self.fc2 = torch.nn.Linear(64, 28)

        self.loss = nn.MSELoss()


    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.pool1(self.batch_norm1(self.relu1(self.conv1(x))))
        x = self.pool2(self.batch_norm2(self.relu2(self.conv2(x))))
        x = self.pool3(self.batch_norm3(self.relu3(self.conv3(x))))
        x = self.pool4(self.batch_norm4(self.relu4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        self.training = True
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        torch.cuda.empty_cache()
        del x, y
        torch.cuda.empty_cache()
        return loss

    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001, weight_decay = 5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min',
                                                                  factor=0.5,
                                                                  patience=10,
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
        self.training = False
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        del x, y
        torch.cuda.empty_cache()
        return loss
def train_detector(train_gt, train_img_dir, fast_train = True):
    transform = A.Compose(
       [A.Rotate(limit=40, p = 0.5),
        A.OneOf([
            A.RGBShift(r_shift_limit=[0, 100], g_shift_limit=[0,100], b_shift_limit=[0,100], p = 0.3),
            A.ToGray(p = 0.5)
        ]),
        A.GaussNoise(p = 0.3),
        A.MotionBlur(p = 0.3),
        ],
        keypoint_params = A.KeypointParams(format='xy')
    )
    
    if not fast_train:
        num_workers = 2
        batch_size = 128
        epochs = 100
        trainer = pl.Trainer(
            max_epochs = epochs,
            accelerator = "gpu",
            callbacks = None,
            logger = False,
            num_sanity_val_steps=0
        )
    else:
        num_workers = 0
        batch_size = 1
        epochs = 1
        trainer = pl.Trainer(
            max_epochs = epochs,
            max_steps=2,
            accelerator = "cpu",
            callbacks = None,
            logger = False,
            enable_checkpointing=False,
            num_sanity_val_steps=0
        )

    ds_train = MyCustomDataset(mode = "train", data_dir = train_img_dir, train_gt = train_gt, transform = transform, store_in_ram = False if fast_train else True)
    ds_val = MyCustomDataset(mode = "val", data_dir = train_img_dir, train_gt = train_gt, transform = transform, store_in_ram = False if fast_train else True)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BaseModel()
    trainer.fit(model, dl_train, dl_val)
    return model

def detect(model_filename, test_img_dir):
    model = BaseModel()
    model.load_state_dict(torch.load(model_filename, map_location=torch.device("cpu")))
    model.eval()
    model.training = False
    model.inference = True
    model.to(device)

    ans = dict()

    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        img, img_shape = load_resize(img_path, [], inference = True)
        img = normalize(img)

        inp = torch.from_numpy(img.transpose(2, 0, 1))[None, :].to(device)

        res = model(inp).detach()

        res[::2] = res[::2] / INPUT_SIZE * img_shape[1]
        res[1::2] = res[1::2] / INPUT_SIZE * img_shape[0]
        res = np.clip(res.cpu().numpy(), 0, max(img_shape[0],img_shape[1])).astype(np.int32)
        ans[img_name] = res.ravel()

    return ans
