import pdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
#import sparseconvnet as scn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall, f1
from typing import Tuple
import numpy as np
from pytorch_lightning import Callback

import copy
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_epoch_end(self, trainer, pl_module):
        metrics = copy.deepcopy(trainer.callback_metrics)
        
        for metric in metrics:
            if metric not in self.metrics:
                self.metrics[metric] = [metrics[metric]]
            self.metrics[metric].append(metrics[metric])
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
                 hidden_layers_in_out = [(512,512), (512,512)]
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
        self.hidden_layers_in_out = hidden_layers_in_out

        self.relu = nn.LeakyReLU() 
        
        if self.use_dropout:
            self.drop=nn.Dropout(p=self.dropout_prob_linear)

        outSize, self.conv_layer1 = self.set_conv_block(inputShape, outChannels, stride)
        # self.conv_layer2 = self.set_conv_block(128, 128)
        self.fc_conv = nn.Linear(outSize, self.hidden_layers_in_out[0][0])
        self.fcs = self.set_fc_hidden_block(self.hidden_layers_in_out)
        self.fc_out = nn.Linear(self.hidden_layers_in_out[-1][-1], self.num_classes)
        
    def set_fc_hidden_block(self,
                            hidden_layers_in_out,
                            ) -> torch.nn.Sequential:
        hidden_layers = []
        for (inNodes, outNodes) in hidden_layers_in_out:
            hidden_layers.append(nn.Linear(inNodes,outNodes))
            hidden_layers.append(nn.LeakyReLU())
            if self.use_dropout:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob_linear))
        fc_block = nn.Sequential(*hidden_layers)
        
        return fc_block

    def set_conv_block(self, 
                       in_shape: Tuple[int, int, int, int],
                       out_c: int,
                       conv_stride: int,
                       conv_kernel_size: int = 6,
                       conv_padding: int = 0,
                       max_stride: int = 2,
                       max_kernel_size: int = 2,
                       max_padding: int = 0,
                       ) -> torch.nn.Sequential:
        '''
        Convolution bulding block
        
        Inputs:
            - in_shape: input shape
            - out_c: number of output channels
            - conv_stride: convolution operation stride
        '''
        layers = []
        # if self.use_batchnorm:
        #     layers.append(nn.BatchNorm3d(in_shape[0]))
       
        layers.append(nn.Conv3d(in_shape[0], out_c, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding))
        # Based on floor((W−F+2P)/S)+1, W = input size, F=kernel/filter size, P=padding
        outputSize = np.floor((np.array(in_shape[1:])-conv_kernel_size+2*conv_padding)/conv_stride)+1

        if self.use_batchnorm:
            layers.append(nn.BatchNorm3d(out_c))
       
        layers.append(self.relu)

        # Now let's do it again for max pool
        layers.append(nn.MaxPool3d(kernel_size=max_kernel_size, padding=max_padding,  stride=max_stride))
        outputSize = np.floor((outputSize-max_kernel_size+2*max_padding)/max_stride)+1
        outputSize = int(np.prod(outputSize)*out_c)
        
        if self.use_dropout:
             layers.append(nn.Dropout3d(p=self.dropout_prob_conv))

        conv_block = nn.Sequential(*layers)

        return outputSize, conv_block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer1(x)
        # out = self.conv_layer2(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc_conv(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.drop(out)
            
        out = self.fcs(out)
            
        out = self.fc_out(out)
        
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

        self.log_dict({
            'loss': loss.mean(),
            'accuracy':accuracy.mean()
        })

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
        
        self.log_dict({
            'valid_loss': loss.mean(),
            'valid_accuracy':accuracy.mean()
        })
        
        return {'loss': loss, 'dict': best_worst_dict}

    '''
    def validation_step_end(self, batch_parts):
        # aggregate when using multiple GPUs
        
        # loss
        losses = batch_parts['loss']
        loss = torch.mean(losses)
        
        # best_worst_dicts
        dicts = batch_parts['dict']
        # find which GPU holded the image with the best score
        best_gpu_idx = torch.argmax(dicts['best'][0])
        max_prob = dicts['best'][0][best_gpu_idx]
        # idxs to select images
        if best_gpu_idx == 0:
            start_idx = 0
            end_edx = 30
        elif best_gpu_idx == 1:
            start_idx = 30
            end_edx = 60
        max_prob_img = dicts['best'][1][start_idx:end_edx, :, :]
        # find which GPU holded the image with the best score
        worst_gpu_idx = torch.argmin(dicts['worst'][0])
        min_prob = dicts['worst'][0][worst_gpu_idx]
        # idxs to select images
        if worst_gpu_idx == 0:
            start_idx = 0
            end_edx = 30
        elif worst_gpu_idx == 1:
            start_idx = 30
            end_edx = 60
        min_prob_img = dicts['worst'][1][start_idx:end_edx, :, :]

        # re-construct return dict
        best_worst_dict = {
            "best"  : [max_prob, max_prob_img],
            "worst" : [min_prob, min_prob_img]
        }
        
        return {'loss': loss, 'dict': best_worst_dict}
    '''

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
        # The first logger is expected to be the default Tensorboard logger.
        self.logger[0].experiment.add_image('high_score_img', self.projection_over_cols(best_img), dataformats='HW')
        self.logger[0].experiment.add_image('low_score_img', self.projection_over_cols(worst_img), dataformats='HW')

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        # clamping very small value to 1e-9 to avoid zero division
        probs = torch.clamp(probs, min=1.0e-9)
        r_hat = (1 - probs) / probs # p_{1}(x) / p_{0}(x) (??) # from: https://github.com/sjiggins/carl-torch/blob/master/ml/evaluate.py#L38
        weights = r_hat # carl-torch inverts this but we don't as we want to re-weight the anternative (p_{1}) -> original (p_{0}). Ref: https://github.com/sjiggins/carl-torch/blob/master/evaluate.py#L67

        return probs, weights


class Conv3DModelGF(Conv3DModel):

    def __init__(self,
                 inputShape: Tuple[int, int, int, int], 
                 num_features: int,
                 learning_rate: float = 0.001, 
                 use_batchnorm: bool = False, 
                 use_dropout: bool = False, 
                 weight_decay: bool = True, 
                 stride: int = 1, 
                 outChannels: int = 6, 
                 hidden_layers_in_out=[(512, 512), (512, 512)]
                 ) -> None:
        super().__init__(inputShape,
                         learning_rate,
                         use_batchnorm, 
                         use_dropout, 
                         weight_decay, 
                         stride, 
                         outChannels, 
                         hidden_layers_in_out)

        outSize, self.conv_layer1 = super().set_conv_block(inputShape, outChannels, stride)
        self.fc_conv = nn.Linear(outSize+num_features, hidden_layers_in_out[0][0])

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer1(image)
        # out = self.conv_layer2(out)
        out = out.view(out.size(0), -1) # flatten
        # concatenate conv output and global features
        out = torch.cat((out, features), dim=1)
        out = self.fc_conv(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.drop(out)
        out = self.fcs(out)
        out = self.fc_out(out)
        
        return out

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, f, y = batch
        logits = self(x, f)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # metrics
        accuracy, precision, recall, f1 = self.calculate_metrics(logits, y)

        self.log('train_loss', loss,            on_step=True,   on_epoch=False, prog_bar=True,  logger=True, sync_dist=True)
        self.log('train_accuracy', accuracy,    on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('train_precision', precision,  on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('train_recall', recall,        on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('train_f1', f1,                on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)

        self.log_dict({
            'loss': loss.mean(),
            'accuracy':accuracy.mean()
        })

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, f, y = batch
        logits = self(x, f)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # metrics
        accuracy, precision, recall, f1 = self.calculate_metrics(logits, y)
        best_worst_dict = self.fetch_best_worst_imgs(logits, x)

        self.log('val_loss', loss,            on_step=True,   on_epoch=False, prog_bar=True,  logger=True, sync_dist=True)
        self.log('val_accuracy', accuracy,    on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('val_precision', precision,  on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('val_recall', recall,        on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        self.log('val_f1', f1,                on_step=False,  on_epoch=True,  prog_bar=False, logger=True, sync_dist=True)
        
        self.log_dict({
            'valid_loss': loss.mean(),
            'valid_accuracy':accuracy.mean()
        })
        
        return {'loss': loss, 'dict': best_worst_dict}

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, f, _ = batch
        logits = self(x, f)
        probs = torch.sigmoid(logits)
        # clamping very small value to 1e-9 to avoid zero division
        probs = torch.clamp(probs, min=1.0e-9)
        r_hat = (1 - probs) / probs # p_{1}(x) / p_{0}(x) (??) # from: https://github.com/sjiggins/carl-torch/blob/master/ml/evaluate.py#L38
        weights = r_hat # carl-torch inverts this but we don't as we want to re-weight the anternative (p_{1}) -> original (p_{0}). Ref: https://github.com/sjiggins/carl-torch/blob/master/evaluate.py#L67

        return probs, weights

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
