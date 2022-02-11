import pdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
#import sparseconvnet as scn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall, f1
from typing import Tuple
import numpy as np

#################################################

class Conv3DModel(pl.LightningModule):
    
    def __init__(self,
                 inputShape: Tuple[int, int, int, int],
                 learning_rate: float = 1e-3,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 weight_decay: bool = True,
                 stride: int = 1,
                 outChannels: int = 6,
                 ) -> None:
        super(Conv3DModel, self).__init__()
        
        self.num_classes = 1
        self.class_one_threshold = 0.5
        self.use_dropout = use_dropout
        self.dropout_prob_conv = 0.20
        self.dropout_prob_linear = 0.50
        self.use_batchnorm = use_batchnorm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.relu = nn.LeakyReLU()
        outSize, self.conv_layer1 = self.set_conv_block(inputShape, outChannels, stride)
        # self.conv_layer2 = self.set_conv_block(128, 128)
        self.fc1 = nn.Linear(outSize, 512) # I still don't know how to calculate the first argument number
        # WH: see below how to calculate it
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        
        self.fc4 = nn.Linear(512, self.num_classes)
        if self.use_dropout:
            self.drop=nn.Dropout(p=self.dropout_prob_linear)
        
    def set_conv_block(self, 
                       in_shape: Tuple[int, int, int, int],
                       out_c: int,
                       conv_stride: int) -> torch.nn.Sequential:
        '''
        Convolution bulding block
        
        Inputs:
            - in_shape: input shape
            - out_c: number of output channels
            - conv_stride: convolution operation stride
        '''
        layers = []
        if self.use_batchnorm:
            layers.append(nn.BatchNorm3d(in_shape[0]))
        kernel_size = 6
        padding = 0
        layers.append(nn.Conv3d(in_shape[0], out_c, kernel_size=kernel_size, stride=conv_stride, padding=padding))
        # Based on floor((W−F+2P)/S)+1, W = input size, F=kernel/filter size, P=padding
        outputSize = np.floor((np.array(in_shape[1:])-kernel_size+2*padding)/conv_stride)+1
        layers.append(self.relu)

        # Now let's do it again for max pool
        kernel_size = 2
        stride = 2
        padding = 0
        layers.append(nn.MaxPool3d(kernel_size=kernel_size, padding=0,  stride=stride))
        outputSize = np.floor((outputSize-kernel_size+2*padding)/stride)+1
        outputSize = int(np.prod(outputSize)*out_c)
        
        if self.use_dropout:
             layers.append(nn.Dropout3d(p=self.dropout_prob_conv))

        conv_block = nn.Sequential(*layers)

        return outputSize, conv_block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer1(x)
        # out = self.conv_layer2(out)
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
        
        return out

    def configure_optimizers(self):
        if self.weight_decay:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=0.005)
        else:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate)

    
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

    def fetch_best_worst_imgs(self, logits: torch.Tensor, imgs: torch.Tensor) -> dict:
        prob = torch.sigmoid(logits)
        # calculate max and min prob
        max_prob = torch.max(prob)
        min_prob = torch.min(prob)
        # calculate max and min prob idx
        max_prob_idx = torch.argmax(prob)
        min_prob_idx = torch.argmin(prob)
        # select the corresponding images
        max_prob_img = imgs[max_prob_idx,0,:,:,:]
        min_prob_img = imgs[min_prob_idx,0,:,:,:]

        out_dict = {
            "best"  : [max_prob, max_prob_img],
            "worst" : [min_prob, min_prob_img]
        }

        return out_dict

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # metrics
        accuracy, precision, recall, f1 = self.calculate_metrics(logits, y)

        self.log('train_loss', loss,            on_step=True,   on_epoch=False, prog_bar=True,  logger=True, sync_dist=True)
        self.log('train_accuracy', accuracy,    on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('train_precision', precision,  on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('train_recall', recall,        on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('train_f1', f1,                on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # metrics
        accuracy, precision, recall, f1 = self.calculate_metrics(logits, y)
        best_worst_dict = self.fetch_best_worst_imgs(logits, x)

        self.log('val_loss', loss,            on_step=True,   on_epoch=False, prog_bar=True,  logger=True, sync_dist=True)
        self.log('val_accuracy', accuracy,    on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('val_precision', precision,  on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('val_recall', recall,        on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('val_f1', f1,                on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)

        return {'loss': loss, 'dict': best_worst_dict}

    # def validation_step_end(self, batch_parts):
    #     # aggregate when using multiple GPUs
        
    #     # loss
    #     losses = batch_parts['loss']
    #     loss = torch.mean(losses)
        
    #     # best_worst_dicts
    #     dicts = batch_parts['dict']
    #     # find which GPU holded the image with the best score
    #     best_gpu_idx = torch.argmax(dicts['best'][0])
    #     max_prob = dicts['best'][0][best_gpu_idx]
    #     # idxs to select images
    #     if best_gpu_idx == 0:
    #         start_idx = 0
    #         end_edx = 30
    #     elif best_gpu_idx == 1:
    #         start_idx = 30
    #         end_edx = 60
    #     max_prob_img = dicts['best'][1][start_idx:end_edx, :, :]
    #     # find which GPU holded the image with the best score
    #     worst_gpu_idx = torch.argmin(dicts['worst'][0])
    #     min_prob = dicts['worst'][0][worst_gpu_idx]
    #     # idxs to select images
    #     if worst_gpu_idx == 0:
    #         start_idx = 0
    #         end_edx = 30
    #     elif worst_gpu_idx == 1:
    #         start_idx = 30
    #         end_edx = 60
    #     min_prob_img = dicts['worst'][1][start_idx:end_edx, :, :]

    #     # re-construct return dict
    #     best_worst_dict = {
    #         "best"  : [max_prob, max_prob_img],
    #         "worst" : [min_prob, min_prob_img]
    #     }
        
    #     return {'loss': loss, 'dict': best_worst_dict}

    def projection_over_cols(self, img):
        return torch.sum(img, 1)
    
    def validation_epoch_end(self, val_step_outputs) -> None:
        # find images with best and worst prob over the validation set
        max_prob = 0
        min_prob = 2
        max_prob_idx = min_prob_idx = -9999

        for i, val_step_output in enumerate(val_step_outputs):
            best_worst_dict = val_step_output['dict']

            best_prob = best_worst_dict['best'][0]
            if best_prob > max_prob:
                max_prob = best_prob
                max_prob_idx = i
            
            worst_prob = best_worst_dict['worst'][0]
            if worst_prob < min_prob:
                min_prob = worst_prob
                min_prob_idx = i
        
        best_img = val_step_outputs[max_prob_idx]['dict']['best'][1]
        worst_img = val_step_outputs[min_prob_idx]['dict']['worst'][1]

        # log images
        self.logger.experiment.add_image('high_score_img', self.projection_over_cols(best_img), dataformats='HW')
        self.logger.experiment.add_image('low_score_img', self.projection_over_cols(worst_img), dataformats='HW')

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        weights = probs / (1 - probs)

        return probs, 1/weights

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
