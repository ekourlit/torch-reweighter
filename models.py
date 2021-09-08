import pdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import sparseconvnet as scn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall, f1
from typing import Tuple

#################################################

class Conv3DModel(pl.LightningModule):
    
    def __init__(self, learning_rate: float, use_batchnorm: bool = False, use_dropout: bool = False) -> None:
        super(Conv3DModel, self).__init__()
        
        self.num_classes = 1
        self.class_one_threshold = 0.5
        self.use_dropout = use_dropout
        self.dropout_prob_conv = 0.20
        self.dropout_prob_linear = 0.50
        self.use_batchnorm = use_batchnorm
        self.learning_rate = learning_rate
        
        self.relu = nn.LeakyReLU()
        self.conv_layer1 = self.set_conv_block(1, 128)
        self.conv_layer2 = self.set_conv_block(128, 128)
        self.fc1 = nn.Linear(27648, 64) # I still don't know how to calculate the first argument number
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.num_classes)
        if self.use_batchnorm:
            self.batch=nn.BatchNorm1d(self.num_hidden_linear_units)
        if self.use_dropout:
            self.drop=nn.Dropout(p=self.dropout_prob_linear)
        
    def set_conv_block(self, in_c: int, out_c: int) -> torch.nn.Sequential:
        layers = []
        layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=0))
        layers.append(self.relu)
        if self.use_batchnorm:
            layers.append(nn.BatchNorm3d(out_c))
        layers.append(nn.MaxPool3d(kernel_size=2))
        if self.use_dropout:
            layers.append(nn.Dropout3d(p=self.dropout_prob_conv))

        conv_block = nn.Sequential(*layers)

        return conv_block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc1(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.drop(out)
        out = self.fc4(out)
        # if self.use_batchnorm:
            # out = self.batch(out)
        
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
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

#################################################

# sparce convolution example for classification
class scnModel(pl.LightningModule):
    def __init__(self):
        super(scnModel, self).__init__()

        self.num_classes = 1
        self.sparseModel = scn.Sequential(

          scn.SparseVggNet(3, 1, [ # second arg seems to be the equivalent of in_c of Conv3d
            ['C', 8], ['C', 8], 'MP',
            ['C', 16], ['C', 16], 'MP',
            ['C', 16, 8], ['C', 16, 8], 'MP',
            ['C', 24, 8], ['C', 24, 8], 'MP' ]),

          scn.Convolution(3, 32, 64, 5, 1, False),

          scn.BatchNormReLU(64),

          scn.SparseToDense(3, 64) )

        self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1, 1]))
        self.inputLayer = scn.InputLayer(3, self.spatial_size, 2) # third arg is batch
        self.linear = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, 64)
        x = self.linear(x)

        return x