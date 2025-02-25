import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models as models

import pytorch_lightning as pl
from torchmetrics.functional.classification import multiclass_accuracy
from keras.applications import ResNet152V2


class ResNet(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, only_pretrained=False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        self.model = models.resnet101(pretrained=True)
        if not only_pretrained:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        print(self.model)

    def _forward_features(self, x):
        x = self.model(x)
        return x

    # will be used during inference
    def forward(self, x):
        #x = F.log_softmax(self.model(x), dim=1)
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = multiclass_accuracy(preds, y, num_classes=self.num_classes)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = multiclass_accuracy(preds, y, num_classes=self.num_classes)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = multiclass_accuracy(preds, y, num_classes=self.num_classes)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
