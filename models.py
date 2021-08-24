import pdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall, f1
from typing import Tuple

#################################################

class Conv3DModel(pl.LightningModule):
    
    def __init__(self, use_batchnorm: bool = False, use_dropout: bool = False) -> None:
        super(Conv3DModel, self).__init__()
        
        self.num_classes = 1
        self.class_one_threshold = 0.5
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.conv_layer1 = self.set_conv_block(1, 32)
        self.conv_layer2 = self.set_conv_block(32, 64)
        self.fc1 = nn.Linear(13824, 128) # I still don't know how to calculate the first argument number
        self.fc2 = nn.Linear(128, self.num_classes)
        self.relu = nn.LeakyReLU()
        if self.use_batchnorm:
            self.batch=nn.BatchNorm1d(128)
        if self.use_dropout:
            self.drop=nn.Dropout(p=0.25)
        
    def set_conv_block(self, in_c: int, out_c: int) -> torch.nn.Sequential:
        conv_block = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=0),
                                   nn.LeakyReLU(),
                                   nn.MaxPool3d(kernel_size=2))
        
        return conv_block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc1(out)
        out = self.relu(out)
        if self.use_batchnorm:
            out = self.batch(out)
        if self.use_dropout:
            out = self.drop(out)
        out = self.fc2(out)
        
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0065)
    
        return optimizer

    def calculate_metrics(self, logits: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate predictions
        prob = torch.sigmoid(logits)
        pred = prob > self.class_one_threshold
        # cast tensors to int
        pred_int = pred.int()
        y_true_int = y_true.int()

        accuracy_score = accuracy(pred_int, y_true_int)
        precision_score = precision(pred_int, y_true_int)
        recall_score = recall(pred_int, y_true_int)
        f1_score = f1(pred_int, y_true_int)

        return accuracy_score, precision_score, recall_score, f1_score

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # metrics
        accuracy, precision, recall, f1 = self.calculate_metrics(logits, y)

        self.log('train_loss', loss,            on_step=True,   on_epoch=False, prog_bar=True,  logger=True)
        self.log('train_accuracy', accuracy,    on_step=False,  on_epoch=True,  prog_bar=False, logger=True)
        self.log('train_precision', precision,  on_step=False,  on_epoch=True,  prog_bar=False, logger=True)
        self.log('train_recall', recall,        on_step=False,  on_epoch=True,  prog_bar=False, logger=True)
        self.log('train_f1', f1,                on_step=False,  on_epoch=True,  prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # metrics
        accuracy, precision, recall, f1 = self.calculate_metrics(logits, y)

        self.log('val_loss', loss,            on_step=True,   on_epoch=False, prog_bar=True,  logger=True)
        self.log('val_accuracy', accuracy,    on_step=False,  on_epoch=True,  prog_bar=False, logger=True)
        self.log('val_precision', precision,  on_step=False,  on_epoch=True,  prog_bar=False, logger=True)
        self.log('val_recall', recall,        on_step=False,  on_epoch=True,  prog_bar=False, logger=True)
        self.log('val_f1', f1,                on_step=False,  on_epoch=True,  prog_bar=False, logger=True)

        return loss

