import os
import shutil
from pathlib import Path
import json
import inspect 
import time
from numpy.core.fromnumeric import transpose
import ruamel.yaml
from logging import Logger, getLogger
from numpy.lib.arraysetops import isin
from shutil import copyfile
import datetime
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn.modules.loss import BCEWithLogitsLoss
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Callback
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_toolbelt.losses import DiceLoss, BinaryFocalLoss, BinaryLovaszLoss, BinarySoftF1Loss
from torchmetrics.functional.classification import binary_accuracy
# from torchmetrics.functional.classification.iou import iou
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.classification import binary_fbeta_score
from torchmetrics.functional.classification import binary_confusion_matrix
try:
    from slumworldML.src.SatelliteDataset import InferenceDataLoader
    from slumworldML.src.base_tiler import ImageTiler
    from slumworldML.src.model import UNet, MODELS_REGISTRY
    from slumworldML.src.utilities import AdaBound, AdaBoundW, Yogi
except Exception as Error:
    try:
        from src.SatelliteDataset import InferenceDataLoader
        from src.base_tiler import ImageTiler
        from src.model import UNet, MODELS_REGISTRY
        from src.utilities import AdaBound, AdaBoundW, Yogi
    except Exception as Error:
        from SatelliteDataset import InferenceDataLoader
        from base_tiler import ImageTiler
        from model import UNet, MODELS_REGISTRY
        from utilities import AdaBound, AdaBoundW, Yogi
import numpy as np
import pandas as pd
import imageio
import logging
import copy
import pdb

logger = logging.getLogger("pytorch_lightning.core")

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

MODEL_PARAMS = { 'in_channels':3, 
                 'out_channels':1, 
                 'features':[64, 128, 256, 512], 
                 'dropout_prob':0.05, 
                 'dropout_2d_prob':0.05,
                 'domain_classifier':False}

TRAINING_PARAMS = {'criterion' :'BinaryCrossEntropyWithLogits',
                   'optimizer' :'SGD',
                   'scheduler':{'name':'Plateau',
                                'params':{'factor': 0.33, 'patience': 5, 'threshold':1e-4},
                                'interval':'epoch'},
                   'val_metric':{'metric': 'val_loss', 'mode': 'min'},
                   'learning_rate' :1e-3,
                   'encoder_learning_rate': None,
                   'final_lr' :0.1,
                   'momentum' :0.9,
                   'enc_dec_learning_rate_ratio': 0.01,
                   'num_epochs':10, 
                   'batch_size' :4,
                   'l2_reg' :1e-5,
                   'threshold':0.5,
                   'tile_size':512,
                   'label_noise': True 
}
class ModelTrainingWrapper(LightningModule):
    '''
    Usage:
        >>> unet_model = UnetTrainer()
        >>> unet_model.defaults
        >>> unet_model.set_model_parameters(**model_params_dict)
        >>> unet_model.set_training_parameters(**training_params_dict)
        >>> trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=20, gpus=1, accumulate_grad_batches=5)
        >>> trainer.fit(unet_model)

    View configurable parameters and their current values:
        >>> unet_model.hparams
    View your logs:
        >>> tensorboard --logdir ./lightning_logs
    '''
    SUPPORTED_LOSSES= {'BinaryCrossEntropyWithLogits':BCEWithLogitsLoss(pos_weight=torch.tensor([2.])),
                       'BinarySoftF1':BinarySoftF1Loss(),
                       'DiceLoss':DiceLoss(mode='binary'), 
                       'BinaryFocalLoss':BinaryFocalLoss(alpha=0.95,gamma=2), 
                       'BinaryLovaszLoss':BinaryLovaszLoss()} 
    SUPPORTED_OPTIMIZERS = { 'Adam':optim.Adam,
                             'AdamW':optim.AdamW,
                             'SGD':optim.SGD,
                             'AdaBound':AdaBound, 
                             'AdaBoundW':AdaBoundW,
                             'Yogi':Yogi
                        }
    SUPPORTED_SCHEDULERS = { 'CosineWarmRestarts':{'name': 'CosineWarmRestarts',
                                                   'scheduler':optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                                   'params':{'T_0':50,'T_mult':2}},
                             'Plateau':{'name':'Plateau',
                                        'scheduler':optim.lr_scheduler.ReduceLROnPlateau,
                                        'params':{'mode': 'min', 'factor': 0.33, 'patience': 5, 'threshold':1e-4}},
                             'Exponential':{'name': 'Exponential',
                                            'scheduler':optim.lr_scheduler.ExponentialLR, 
                                            'params':{'gamma':0.95}},
                             'OneCycleLR':{'name': 'OneCycle',
                                        'scheduler':optim.lr_scheduler.OneCycleLR, 
                                        'params':{'max_lr':None, 'steps_per_epoch':100, 'eta_min':1e-7}},
                             'MultiStepLR': {'name':'MultiStepLR',
                                             'scheduler': optim.lr_scheduler.MultiStepLR,
                                             'params':{'milestones':[50,100,200],
                                             'gamma':0.2}}
                        }
    def __init__(self, model= None, model_params=None, training_params=None, domain_loss_scaling_factor=6e+4,
                from_pretrained=None, training_mode='train_all', domain_classifier=False, logger=None):
        super(ModelTrainingWrapper, self).__init__()
        self.from_pretrained = from_pretrained
        self.classifier = domain_classifier
        self.domain_loss_scaling_factor = domain_loss_scaling_factor
        if from_pretrained:
            print(f"Initializing model '{from_pretrained}' from registry.")
            model_class = MODELS_REGISTRY[from_pretrained]
            params_for_model = model_params if model_params is not None else MODEL_PARAMS
            params_for_model['train_head_only'] = (training_mode == 'freeze')
            params_for_model['domain_classifier'] = domain_classifier
            valid_keys = inspect.signature(model_class.__init__).parameters.keys()
            filtered_params = {k: v for k, v in params_for_model.items() if k in valid_keys}
            # this assumes the key must be completely equal
            print(f"Passing filtered parameters to model: {filtered_params}")
            self.model = model_class(**filtered_params)
        
        elif model is not None:
            self.model = model
        else:
            print("No pretrained model or model instance provided, defaulting to vanilla UNet.")
            self.model = UNet(**(model_params or MODEL_PARAMS))

        self.training_mode = training_mode
        self.set_training_parameters(**(training_params or TRAINING_PARAMS))

        self.custom_logs = {'train_loss':[0.0,], 'train_acc':[0.0,], 'val_loss':[], 'val_acc':[], 
                            'val_f1':[], 'val_iou':[], 'val_conf_mat':[]}
        if self.classifier:
            domain_logs = {'train_domain_loss':[0.0,], 'train_domain_acc':[0.0,], 'val_domain_loss':[], 'val_domain_acc':[] }
            self.custom_logs.update(domain_logs)
            
        print(f"\n\nL2 regularizer:{self.l2_reg}\n\n")

    def set_loss_function(self, loss_fn):
        assert loss_fn in list(self.SUPPORTED_LOSSES.keys()), "Loss function not supported. Select one of {}".format(self.SUPPORTED_LOSSES.keys())
        self.criterion = self.SUPPORTED_LOSSES[loss_fn]
        self.hparams['criterion'] = loss_fn
        if self.classifier:
            self.classification_criterion = BCEWithLogitsLoss()
    
    def set_optimizer(self, optimizer):
        assert optimizer in list(self.SUPPORTED_OPTIMIZERS.keys()), "Optimizer not supported. Select one of {}".format(self.SUPPORTED_OPTIMIZERS.keys())
        decoder_lr = self.learning_rate
        encoder_lr = getattr(self, 'encoder_learning_rate', decoder_lr)
        if encoder_lr is None: encoder_lr = decoder_lr

        if hasattr(self.model, 'encoder') and encoder_lr != decoder_lr:
            print(f"[Optimizer] Dual Learning Rate Strategy Enabled.")
            print(f"  - Encoder LR: {encoder_lr}")
            print(f"  - Decoder/Head LR: {decoder_lr}")

            encoder_param_ids = list(map(id, self.model.encoder.parameters()))
            encoder_params = self.model.encoder.parameters()
            decoder_params = filter(lambda p: id(p) not in encoder_param_ids, self.model.parameters())
            parameters = [
                {'params': decoder_params, 'lr': decoder_lr},  # Group 0
                {'params': encoder_params, 'lr': encoder_lr}  # Group 1
            ]
        else:
            if encoder_lr != decoder_lr:
                print(
                    "[Optimizer] Warning: Different LRs specified but 'encoder' module not found in model. Using global LR.")

            print(f"[Optimizer] Using Global LR: {decoder_lr}")
            parameters = self.model.parameters()

        if optimizer in ['Adam', 'AdamW']:
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer](parameters, lr=decoder_lr, weight_decay=self.l2_reg)
        elif optimizer in ['SGD']:
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer](parameters, lr=decoder_lr, momentum=self.momentum,
                                                                  weight_decay=self.l2_reg)
        elif optimizer in ['AdaBound', 'AdaBoundW']:
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer](parameters, lr=decoder_lr, final_lr=self.final_lr,
                                                                  weight_decay=self.l2_reg)
        elif optimizer == 'Yogi':
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer](parameters, lr=decoder_lr, weight_decay=self.l2_reg)

        self.hparams['optimizer'] = optimizer

    def set_scheduler(self, scheduler):
        if scheduler['name'] is not None:
            assert scheduler['name'] in list(self.SUPPORTED_SCHEDULERS.keys()), "Scheduler not supported. Select one of {}".format(self.SUPPORTED_SCHEDULERS.keys())
            if scheduler['name'] != "OneCycleLR":
                self.scheduler = self.SUPPORTED_SCHEDULERS[scheduler['name']]['scheduler'](optimizer=self.optimizer, **scheduler['params'])
            else:
                self.scheduler = self.SUPPORTED_SCHEDULERS[scheduler['name']]['scheduler'](optimizer=self.optimizer, steps_per_epoch=scheduler['params']['steps_per_epoch'],
                                                                                           epochs=self.num_epochs, max_lr=scheduler['params']['max_lr'],)
        else:
            self.scheduler = None
        self.hparams['scheduler'] = scheduler

    def set_training_parameters(self, **kwargs):
        for param, val in kwargs.items():
            if param not in list(TRAINING_PARAMS.keys()):
                print("Parameter {} not understood. Will be ignored.".format(param))
                continue
            if param not in ['optimizer', 'criterion', 'scheduler']:
                if TRAINING_PARAMS[param] is None or isinstance(val, type(TRAINING_PARAMS[param])):
                    setattr(self, param, val)
                    self.hparams[param] = val
                else:
                    print(f"Parameter {param} not of the correct type.")
                    print(f"Expected {type(TRAINING_PARAMS[param])}, got {type(val)}. Default value:{TRAINING_PARAMS[param]}.")
                    continue
        if 'optimizer' in kwargs.keys():
            self.set_optimizer(kwargs['optimizer'])
        if 'scheduler' in kwargs.keys():
            self.set_scheduler(kwargs['scheduler'])
        if 'criterion' in kwargs.keys():
            self.set_loss_function(kwargs['criterion'])

    def set_model_parameters(self, update_model=False, **kwargs):
        for param, val in kwargs.items():
            if param not in list(MODEL_PARAMS.keys()):
                print("Parameter {} not understood. Will be ignored.".format(param))
                continue
            if not isinstance(val, type(MODEL_PARAMS[param])):
                print(f"Parameter {param} not of the correct type.")
                print(f"Expected {type(MODEL_PARAMS[param])}, got {type(val)}. Default value:{MODEL_PARAMS[param]}.")
                continue
            if update_model:
                setattr(self.model, param, val)
            self.hparams['model'][param] = val

    def forward(self, image, dino_features=None):
        # This now directly calls the internal model `self.model` which expects two arguments.
        if dino_features is None:
            return self.model(image)
        try:
            return self.model(image, dino_features)
        except TypeError as e:
            msg = str(e)
            # secondary handle
            if ("positional argument" in msg and "given" in msg) or ("takes" in msg and "given" in msg):
                return self.model(image)
            raise
        # output = self.model(x)
        # if len(output) == 2: # we have a domain classifier
        #     return output[0], output[1]
        # else:
        #     return output

    def configure_optimizers(self):
        optimizer = self.optimizer 
        if self.scheduler is not None:
            scheduler_name = self.hparams['scheduler']['name']
            lr_scheduler = {} 

            if scheduler_name == 'Plateau': 
                lr_scheduler = {
                    'scheduler': self.scheduler,
                    'monitor': self.val_metric['metric'], 
                    'interval': self.hparams['scheduler']['interval'],
                    'frequency': 1, 
                }
            elif scheduler_name == 'OneCycleLR': 
                lr_scheduler = {
                    'scheduler': self.scheduler,
                    'interval': self.hparams['scheduler']['interval'], 
                    'frequency': 1,
                }
            else: 
                lr_scheduler = {
                    'scheduler': self.scheduler,
                    'interval': self.hparams['scheduler']['interval'],
                    'frequency': 1,
                }

            return [optimizer], [lr_scheduler] 
        else:
            return optimizer

    @torch.no_grad()
    def _calc_metrics(self, labels, outputs):
        # Convert raw model outputs (logits) to probabilities and then to binary predictions.
        preds = (torch.sigmoid(outputs) > self.threshold).long() # Use .long() for torchmetrics
        hard_labels = (labels > 0.5).long()
        acc = binary_accuracy(preds=preds, target=hard_labels)
        f1_score = binary_fbeta_score(preds=preds, target=hard_labels, beta=1.0)
        iou_score = binary_jaccard_index(preds=preds, target=hard_labels) #handles 0/0 type error
        conf_mat = torch.flip(binary_confusion_matrix(preds=preds, target=hard_labels), [0,1])
        pbar = {'accuracy': acc, 
                'f1-score':f1_score, 
                'iou': iou_score ,
                'confusion-matrix':conf_mat}
        return pbar

    def _describe_structure(self, obj, depth=0):
        indent = '  ' * depth
        if isinstance(obj, (list, tuple)):
            desc = f"{type(obj).__name__} (len={len(obj)})"
            if len(obj) > 0:
                sample_desc = ModelTrainingWrapper._describe_structure(obj[0], depth+1)
                return f"{desc}\n{indent}└─ {sample_desc}"
            return desc
        elif isinstance(obj, dict):
            desc = f"dict (keys: {list(obj.keys())})"
            for i, (k, v) in enumerate(obj.items()):
                if i >= 3: 
                    desc += f"\n{indent}└─ ... ({len(obj)-3} more keys)"
                    break
                branch = '├─' if i < len(obj)-1 else '└─'
                item_desc = ModelTrainingWrapper._describe_structure(v, depth+1)
                desc += f"\n{indent}{branch} {k}: {item_desc}"
            return desc
        elif hasattr(obj, 'shape'):
            return f"{type(obj).__name__} {obj.shape}"
        else:
            return f"{type(obj).__name__}"

    def on_train_start(self):
        super().on_train_start() # Call parent method
        print(f"DEBUG: on_train_start called. current_epoch: {self.current_epoch}")
        # Check if self.trainer is available and has loggers
        if hasattr(self, 'trainer') and self.trainer is not None and self.trainer.loggers:
            loggers = self.trainer.loggers if isinstance(self.trainer.loggers, list) else [self.trainer.loggers]
            for logger_instance in loggers:
                if isinstance(logger_instance, (pl.loggers.CSVLogger, pl.loggers.TensorBoardLogger)):
                    print(f"DEBUG: Logger type: {type(logger_instance).__name__}, log_dir: {logger_instance.log_dir}")
                    if hasattr(logger_instance, 'log_hyperparams'):
                        logger_instance.log_hyperparams({"hp/loss":0.,"hp/acc": 0., "hp/f1": 0., "hp/iou":0.})
                    else:
                        print(f"DEBUG: Logger {type(logger_instance).__name__} does not have log_hyperparams method.")
                else:
                    print(f"DEBUG: Non-standard PL logger found: {type(logger_instance).__name__}")
        else:
            print("DEBUG: Trainer or loggers not available at on_train_start.")

        if self.training_mode == "fine_tune":
            try:
                self.model._set_trainable(train_head_only=False)
            except AttributeError:
                print("Warning! Could not find _set_trainable method in model.")


    

    # def on_validation_epoch_start(self):
    #     super().on_validation_epoch_start() 

    #     if hasattr(self, 'trainer') and self.trainer is not None and \
    #        hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
    #         print(f"DEBUG_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: Forcing fresh val_dataloader.")
            
    #         self.trainer.val_dataloaders = self.trainer.datamodule.val_dataloader()
            
    #         if isinstance(self.trainer.val_dataloaders, (list, tuple)) and len(self.trainer.val_dataloaders) > 0:
    #             actual_combined_loader = self.trainer.val_dataloaders[0] 
    #         else:
    #             # ensure val_dataloader iteratble, maynot happen but for debugging use
    #             print(f"ERROR_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: val_dataloaders not a list or empty after refresh.")
    #             return 

    #         try:
    #             target_dataloader = actual_combined_loader.loaders['slum_prediction'] 
    #             test_batch = next(iter(target_dataloader))
                
    #             if isinstance(test_batch, dict):
    #                 print(f"DEBUG_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: Successfully pulled first batch from forced val_dataloader. Batch keys: {list(test_batch.keys())}")
    #             elif isinstance(test_batch, (list, tuple)):
    #                 print(f"DEBUG_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: Successfully pulled first batch (tuple type). Input shape: {test_batch[0].shape}, Label shape: {test_batch[1].shape}, Paths count: {len(test_batch[2])}")
    #             else:
    #                 print(f"DEBUG_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: Successfully pulled first batch, but unexpected type: {type(test_batch)}")

    #         except StopIteration:
    #             print(f"DEBUG_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: Forced val_dataloader is empty at start of epoch. THIS IS A PROBLEM.")
    #         except Exception as e:
    #             # debugging
    #             print(f"ERROR_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: Error pulling batch from forced val_dataloader: {e}")
    #             print(f"DEBUG_MTW_FORCE: Type of self.trainer.val_dataloaders (after refresh): {type(self.trainer.val_dataloaders)}")
    #             if isinstance(self.trainer.val_dataloaders, (list, tuple)) and len(self.trainer.val_dataloaders) > 0:
    #                 print(f"DEBUG_MTW_FORCE: Type of self.trainer.val_dataloaders[0]: {type(self.trainer.val_dataloaders[0])}")
    #                 if hasattr(self.trainer.val_dataloaders[0], 'loaders'):
    #                     print(f"DEBUG_MTW_FORCE: Keys/Structure of self.trainer.val_dataloaders[0].loaders: {self.trainer.val_dataloaders[0].loaders.keys()}")
    #     else:
    #         print(f"DEBUG_MTW_FORCE: Epoch {self.current_epoch} - on_validation_epoch_start: trainer or datamodule not available.")
    

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            # Case 1: Using CombinedLoader.
            # The 'slum_prediction' key is always expected.
            images, dino_features, labels, paths = batch['slum_prediction']
            # if fine-tuning
            if 'target_finetuning' in batch:
                ft_images, ft_dino, ft_labels, ft_paths = batch['target_finetuning']
                ft_images = ft_images.to(images.device)
                ft_dino = ft_dino.to(dino_features.device)
                ft_labels = ft_labels.to(labels.device)
                #In-Batch Mixing
                images = torch.cat([images, ft_images], dim=0)
                dino_features = torch.cat([dino_features, ft_dino], dim=0)
                labels = torch.cat([labels, ft_labels], dim=0)
            # if domain adaptation
            if self.classifier and 'domain_prediction' in batch:
                adapt_images, adapt_dino_features, _, adapt_names = batch['domain_prediction']
            else:
                adapt_images, adapt_dino_features, adapt_names = None, None, None

        # elif isinstance(batch, (list, tuple)) and len(batch) == 4:
        #     # Case 2: Using a standard DataLoader (no domain adaptation).
        #     images, dino_features, labels, paths = batch
        #     adapt_images, adapt_dino_features, adapt_names = None, None, None
        else:
            raise ValueError(f"Unsupported batch format in training_step (trainer.py). Expected Type: Dict. Received type: {type(batch)}")

        labels = labels.to(dtype=images.dtype)
        #get the row length of the image after concat finished
        batch_size_1 = images.shape[0]
        
        # Now, check if adapt_images is not None before using it.
        if self.classifier and adapt_images is not None:
            batch_size_2 = len(adapt_names)
            domain_labels = torch.cat([torch.zeros([batch_size_1, 1], device=adapt_images.device), torch.ones([batch_size_2, 1],device=adapt_images.device)])
            domain_labels = domain_labels.to(dtype=adapt_images.dtype).to(self.device)
            
            images_ = torch.cat((images, adapt_images), axis=0)
            dino_features_ = torch.cat((dino_features, adapt_dino_features), axis=0)
            
            outputs_, logits = self(images_, dino_features_)
        else: 
            outputs_ = self(images, dino_features)

        outputs = outputs_[:batch_size_1]
        labels = labels.view(outputs.shape)
        loss = self.criterion(outputs, labels)

        if self.classifier and adapt_images is not None:
            classification_loss = self.domain_loss_scaling_factor * self.classification_criterion(logits, domain_labels)
            domain_labels = domain_labels.view(logits.shape)
            loss += classification_loss

        with torch.no_grad():
            acc = binary_accuracy(preds=(torch.sigmoid(outputs.view((outputs.shape[0],-1))) > self.threshold).float(), 
                              target=(labels.view((outputs.shape[0],-1))>0.5).long())
            train_stats = {'loss':{'train':loss.cpu().detach().item()},
                           'accuracy':{'train':acc.cpu().detach().item()}}
            if self.classifier and adapt_images is not None:
                domain_acc = binary_accuracy(preds=(torch.sigmoid(logits.view((logits.shape[0],-1))) > 0.5).float(), 
                                target=(domain_labels.view((logits.shape[0],-1))>0.5).long())
                domain_stats = { 'domain_loss': {'train':classification_loss.cpu().detach().item()},
                                 'domain_accuracy':{'train':domain_acc.cpu().detach().item()}}
                train_stats.update(domain_stats)

        return {'loss':loss, 'log':train_stats, 'progress_bar':train_stats}


    def training_step_end(self, losses):
        '''agregate across gpus (data parallel mode)'''
        if isinstance(losses['loss'], list):
            train_stats = { 'loss':{'train':np.mean([tl['log']['loss']['train'] for tl in losses])},
                            'accuracy':{'train':np.mean([tl['log']['accuracy']['train'] for tl in losses])}, 
                            }
            if self.classifier:
                domain_stats = { 'domain_loss': {'train':np.mean([tl['log']['domain_loss']['train'] for tl in losses])},
                                 'domain_accuracy':{'train':np.mean([tl['log']['domain_accuracy']['train'] for tl in losses])}
                                }
                train_stats.update(domain_stats)

            loss_dict = {'loss':torch.stack([l['loss'] for l in losses]).mean(), 'progress_bar':train_stats['accuracy']['train'], 'log':train_stats}
        else:
            # In DP mode, DataParallel gathers per-GPU losses into shape [n_gpus].
            # backward() requires a scalar, so reduce with mean().
            loss_dict = losses
            if isinstance(loss_dict.get('loss'), torch.Tensor) and loss_dict['loss'].numel() > 1:
                loss_dict = dict(loss_dict)
                loss_dict['loss'] = loss_dict['loss'].float().mean()
        return loss_dict

    def training_epoch_end(self, loss_steps):
        '''agregate across batches'''
        if isinstance(loss_steps, list):
            # implementation for data parallel mode
            def _to_float(v):
                # handles Python float (single GPU), CPU/CUDA scalar tensor, or DP-gathered [n_gpus] tensor
                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().float().mean().item()
                return float(v)

            train_stats = { 'loss':{'train':np.mean([_to_float(tl['log']['loss']['train']) for tl in loss_steps])},
                            'accuracy':{'train':np.mean([_to_float(tl['log']['accuracy']['train']) for tl in loss_steps])}
                            }
            if self.classifier:
                domain_stats = { 'domain_loss':{'train':np.mean([_to_float(tl['log']['domain_loss']['train']) for tl in loss_steps])},
                                 'domain_accuracy':{'train':np.mean([_to_float(tl['log']['domain_accuracy']['train']) for tl in loss_steps])}
                                }
                train_stats.update(domain_stats)
            def _to_loss_tensor(v):
                if isinstance(v, torch.Tensor):
                    return v.float().mean()
                return torch.tensor(float(v))
            loss_dict = {'loss':torch.stack([_to_loss_tensor(l['loss']) for l in loss_steps]).mean(), 'progress_bar':train_stats['accuracy']['train'], 'log':train_stats}
        else:
            loss_dict = loss_steps
        self.log('train_loss',loss_dict['log']['loss']['train'],prog_bar=True)
        self.log('train_acc', train_stats['accuracy']['train'], prog_bar=True)
        self.custom_logs['train_loss'].append(train_stats['loss']['train'])
        self.custom_logs['train_acc'].append(train_stats['accuracy']['train'])

        #Debugging
        print(f"DEBUG: Epoch {self.current_epoch} - training_epoch_end: train_loss_len={len(self.custom_logs['train_loss'])}, train_acc_len={len(self.custom_logs['train_acc'])}")
        print(f"DEBUG: Epoch {self.current_epoch} - last train_loss: {self.custom_logs['train_loss'][-1]:.4f}, last train_acc: {self.custom_logs['train_acc'][-1]:.4f}")

        if self.classifier:
            self.log('train_domain_loss',loss_dict['log']['domain_loss']['train'],prog_bar=True)
            self.log('train_domain_acc', train_stats['domain_accuracy']['train'], prog_bar=True)
            self.custom_logs['train_domain_loss'].append(train_stats['domain_loss']['train'])
            self.custom_logs['train_domain_acc'].append(train_stats['domain_accuracy']['train'])

    def validation_step(self, batch, batch_idx):
        import threading
        _tid = threading.current_thread().name
        print(f"DEBUG: validation_step START batch_idx={batch_idx} thread={_tid}")
        try:
            return self._validation_step_impl(batch, batch_idx)
        except Exception as _e:
            import traceback as _tb
            print(f"DEBUG: validation_step EXCEPTION thread={_tid}: {type(_e).__name__}: {_e}")
            _tb.print_exc()
            raise

    def _validation_step_impl(self, batch, batch_idx):
        import threading
        _tid = threading.current_thread().name
        # This handles batches that are wrapped in an extra list by Lightning.
        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], (dict, list, tuple)):
            batch = batch[0]

        # This robust logic handles both CombinedLoader (dict) and standard DataLoader (list/tuple)
        images, dino_features, labels, paths = None, None, None, None
        adapt_images, adapt_dino_features, adapt_names = None, None, None

        if isinstance(batch, dict):
            if 'slum_prediction' in batch:
                images, dino_features, labels, paths = batch['slum_prediction']
            if 'target_finetuning' in batch:
                ft_images, ft_dino, ft_labels, ft_paths = batch['target_finetuning']
                ft_images = ft_images.to(ft_images.device)
                ft_dino = ft_dino.to(ft_images.device)
                ft_labels = ft_labels.to(ft_images.device)
                # concat
                if images is None:
                    images, dino_features, labels, paths = ft_images, ft_dino, ft_labels, ft_paths
                else:
                    images = torch.cat([images, ft_images], dim=0)
                    dino_features = torch.cat([dino_features, ft_dino], dim=0)
                    labels = torch.cat([labels, ft_labels], dim=0)

            if self.classifier and 'domain_prediction' in batch:
                adapt_images, adapt_dino_features, _, adapt_names = batch['domain_prediction']

        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            images, dino_features, labels, paths = batch
        else:
            raise ValueError(f"Unsupported batch format in validation_step. Received type: {type(batch)}")

        if images is None:
            raise ValueError("Validation Step: No labeled data found in batch!")

        # In DP mode, DataParallel scatters inputs to each replica's device before calling
        # validation_step, so images.device is the correct device for this replica.
        # next(self.model.parameters()).device would raise StopIteration on GPU 1+ replicas
        # because self.model has no parameters in the shallow-copied DP replica.
        _device = images.device
        print(f"DEBUG: validation_step device={_device}, thread={_tid}, batch_idx={batch_idx}")
        images = images.to(_device)
        dino_features = dino_features.to(_device)
        labels = labels.to(_device, dtype=images.dtype)
        # get concatenated batch size
        batch_size_1 = images.shape[0]

        #deal with adaptation and send to the model
        if self.classifier and adapt_images is not None:
            batch_size_2 = len(adapt_names)
            adapt_images = adapt_images.to(_device)
            adapt_dino_features = adapt_dino_features.to(_device)
            domain_labels = torch.cat([
                torch.zeros([batch_size_1, 1], device=_device),
                torch.ones([batch_size_2, 1], device=_device)
            ])
            domain_labels = domain_labels.to(dtype=images.dtype)

            images_ = torch.cat((images, adapt_images), axis=0)
            dino_features_ = torch.cat((dino_features, adapt_dino_features), axis=0)

            outputs_, logits = self(images_, dino_features_)
            domain_labels = domain_labels.view(logits.shape)
        else:
            # feed into the model
            print(f"DEBUG: validation_step BEFORE forward, thread={_tid}, device={_device}, images.shape={images.shape}")
            outputs_ = self(images, dino_features)
            print(f"DEBUG: validation_step AFTER forward, thread={_tid}")

        outputs = outputs_[:batch_size_1]
        labels = labels.view(outputs.shape)

        #loss calc
        with torch.no_grad():
            # segmentation Loss
            loss = self.criterion(outputs, labels)
            classification_loss = 0.0

            # domain Loss 
            if self.classifier and adapt_images is not None:
                classification_loss = self.domain_loss_scaling_factor * self.classification_criterion(logits,
                                                                                                      domain_labels)
                loss += classification_loss

        pbar = self._calc_metrics((labels > 0.5).long(), outputs)

        pbar['loss'] = loss

        val_stats = {
            'loss': {'val': loss}, 
            'accuracy': {'val': pbar['accuracy']},
            'f1-score': {'val': pbar['f1-score']},
            'iou': {'val': pbar['iou']},
            'confusion-matrix': {'val': pbar['confusion-matrix']},
        }

        if self.classifier and adapt_images is not None:
            pbar['domain_loss'] = classification_loss
            pbar['domain_accuracy'] = binary_accuracy(
                preds=(torch.sigmoid(logits.view((logits.shape[0], -1))) > 0.5).float(),
                target=(domain_labels.view((logits.shape[0], -1)) > 0.5).long()
            )
            domain_stats = {
                'domain_loss': {'val': classification_loss},
                'domain_accuracy': {'val': pbar['domain_accuracy']}
            }
            val_stats.update(domain_stats)

        print(f"DEBUG: validation_step END batch_idx={batch_idx}, thread={_tid}, loss={loss.item():.4f}")
        return {'loss': loss, 'log': val_stats, 'progress_bar': val_stats, 'conf_mat_tensor': pbar['confusion-matrix']}

    def validation_step_end(self, val_step_outputs):
        import traceback as _tb
        print(f"DEBUG: validation_step_end CALLED. isinstance(list)={isinstance(val_step_outputs, list)}, type={type(val_step_outputs).__name__}")
        try:
            return self._validation_step_end_impl(val_step_outputs)
        except Exception as _e:
            print(f"DEBUG: validation_step_end EXCEPTION {type(_e).__name__}: {_e}")
            _tb.print_exc()
            raise

    def _validation_step_end_impl(self, val_step_outputs):
        '''agreggate across gpus (data parallel mode)'''
        if isinstance(val_step_outputs, list):
            val_loss = torch.tensor([x['log']['loss']['val'].detach() for x in val_step_outputs]).mean()
            val_acc = torch.tensor([x['log']['accuracy']['val'].detach() for x in val_step_outputs]).mean()
            val_iou = torch.tensor([x['log']['iou']['val'].detach() for x in val_step_outputs]).mean()
            val_f1_score = torch.tensor([x['log']['f1-score']['val'].detach() for x in val_step_outputs]).mean()
            val_conf_mat = torch.stack([x['log']['confusion-matrix']['val'].detach() for x in val_step_outputs]).sum(0)
            val_stats =  { 'loss':{'val':val_loss}, 'accuracy':{'val':val_acc},
                            'iou':{'val':val_iou}, 'f1-score':{'val':val_f1_score},
                            'confusion-matrix':{'val':val_conf_mat},
                        }
            self.log("hp/loss", val_loss)
            self.log("hp/acc", val_acc)
            self.log("hp/f1", val_f1_score)
            self.log("hp/iou", val_iou)
            if self.classifier:
                val_domain_loss = torch.tensor([x['log']['domain_loss']['val'].detach() for x in val_step_outputs]).mean()
                val_domain_acc = torch.tensor([x['log']['domain_accuracy']['val'].detach() for x in val_step_outputs]).mean()
                domain_stats = {'domain_loss':{'val':val_domain_loss}, 'domain_accuracy':{'val':val_domain_acc}}
                self.log("hp/domain_loss", val_domain_loss)
                self.log("hp/domain_acc", val_domain_acc)
                
            total_loss = torch.tensor([x['loss'].detach() for x in val_step_outputs]).mean()
            return {'loss':total_loss, 'log':val_stats, 'progress_bar':val_stats['iou']['val']}
        else:
            # In DP mode, DataParallel.gather unsqueezes 0-dim tensors to shape [n_gpus].
            # Use .float().mean() to collapse back to a scalar before self.log() (which requires numel==1).
            self.log("hp/loss", val_step_outputs['log']['loss']['val'].detach().float().mean())
            self.log("hp/acc", val_step_outputs['log']['accuracy']['val'].detach().float().mean())
            self.log("hp/f1", val_step_outputs['log']['f1-score']['val'].detach().float().mean())
            self.log("hp/iou", val_step_outputs['log']['iou']['val'].detach().float().mean())
            if self.classifier:
                self.log("hp/domain_loss", val_step_outputs['log']['domain_loss']['val'].detach().float().mean())
                self.log("hp/domain_acc", val_step_outputs['log']['domain_accuracy']['val'].detach().float().mean())
            return val_step_outputs

    def validation_epoch_end(self, val_step_outputs):
        print(f"DEBUG: Epoch {self.current_epoch} - validation_epoch_end: called. len(val_step_outputs) = {len(val_step_outputs)}")
        if val_step_outputs:
            print(f"DEBUG: val_step_outputs[0] type={type(val_step_outputs[0]).__name__}, keys={list(val_step_outputs[0].keys()) if isinstance(val_step_outputs[0], dict) else 'N/A'}")
        if not val_step_outputs:
            # Log sentinel values so early stopping / callbacks don't crash on missing metric
            self.log('val_loss', float('nan'), prog_bar=True)
            self.log('val_acc', float('nan'), prog_bar=True)
            self.log('val_iou', float('nan'), prog_bar=True)
            self.log('val_f1', float('nan'), prog_bar=True)
            return
        '''agreggate across batches'''
        # .float().mean() handles both scalar (single GPU) and shape [n_gpus] (DP gathered) tensors
        mean_val_loss = torch.stack([x['log']['loss']['val'].float().mean() for x in val_step_outputs]).cpu().mean().item()
        mean_val_acc = torch.stack([x['log']['accuracy']['val'].float().mean() for x in val_step_outputs]).cpu().mean().item()
        # In DP mode, confusion matrix is gathered along dim=0: [2,2] per GPU → [n_gpus*2, 2] after gather.
        # .view(-1, 2, 2).sum(0) works for both single GPU ([2,2]→[1,2,2].sum→[2,2]) and DP ([4,2]→[2,2,2].sum→[2,2]).
        total_conf_mat = torch.stack([x['log']['confusion-matrix']['val'].view(-1, 2, 2).sum(0) for x in val_step_outputs]).sum(0)
        # Calculate metrics from the aggregated matrix
        # Assuming format is [[TP, FN], [FP, TN]] based on your code
        tp = total_conf_mat[0, 0].item()
        fn = total_conf_mat[0, 1].item()
        fp = total_conf_mat[1, 0].item()
        tn = total_conf_mat[1, 1].item()

        # Both IoU and F1 are derived from the same global confusion matrix to ensure
        # consistent (micro) aggregation. Averaging per-batch scores would deflate F1
        # on batches with no positive pixels (binary_fbeta_score returns 0) while
        # leaving the confusion-matrix IoU unaffected, creating a spurious IoU > F1.
        iou_denom = tp + fp + fn
        mean_val_iou = tp / iou_denom if iou_denom > 0 else 0.0
        f1_denom = 2 * tp + fp + fn
        mean_val_f1_score = (2 * tp / f1_denom) if f1_denom > 0 else 0.0
        mean_val_conf_mat = torch.stack([x['log']['confusion-matrix']['val'].view(-1, 2, 2).sum(0) for x in val_step_outputs]).sum(0).clone().cpu().detach().numpy().tolist()
        self.custom_logs['val_loss'].append(mean_val_loss)
        self.custom_logs['val_acc'].append(mean_val_acc)
        self.custom_logs['val_iou'].append(mean_val_iou)
        self.custom_logs['val_f1'].append(mean_val_f1_score)
        self.custom_logs['val_conf_mat'].append(mean_val_conf_mat)

        # === Debugging ===
        print(f"DEBUG: Epoch {self.current_epoch} - validation_epoch_end: val_loss_len={len(self.custom_logs['val_loss'])}, val_acc_len={len(self.custom_logs['val_acc'])}")
        print(f"DEBUG: Epoch {self.current_epoch} - last val_loss: {self.custom_logs['val_loss'][-1]:.4f}, last val_acc: {self.custom_logs['val_acc'][-1]:.4f}, last val_f1: {self.custom_logs['val_f1'][-1]:.4f}")
        if hasattr(self, 'trainer') and self.trainer is not None:
            if self.trainer.loggers:
                print(f"DEBUG: Epoch {self.current_epoch} - Loggers present: {', '.join([type(l).__name__ for l in self.trainer.loggers])}")
                for logger_instance in self.trainer.loggers:
                    print(f"DEBUG: Epoch {self.current_epoch} - Logger ({type(logger_instance).__name__}) log_dir: {logger_instance.log_dir}")
            else:
                print(f"DEBUG: Epoch {self.current_epoch} - No loggers configured for Trainer.")

        if hasattr(self, 'trainer') and self.trainer is not None and \
           hasattr(self.trainer, 'logger_connector') and self.trainer.logger_connector is not None and \
           hasattr(self.trainer.logger_connector, 'results') and self.trainer.logger_connector.results is not None:
            
            lightning_logged_val_loss_key = f"validation_step.val_loss" 
            if f"validation_step.val_loss.0" in self.trainer.logger_connector.results: 
                 lightning_logged_val_loss_key = f"validation_step.val_loss.0" 
            elif f"val_loss" in self.trainer.logger_connector.results: 
                 lightning_logged_val_loss_key = f"val_loss"

            if lightning_logged_val_loss_key in self.trainer.logger_connector.results:
                val_loss_metric = self.trainer.logger_connector.results[lightning_logged_val_loss_key]
                
                if isinstance(val_loss_metric, ResultMetricCollection):
                    if 'slum_prediction' in val_loss_metric:
                        val_loss_metric = val_loss_metric['slum_prediction']
                    elif 'val_loss' in val_loss_metric: 
                        val_loss_metric = val_loss_metric['val_loss']
                    else:
                        print(f"DEBUG: Epoch {self.current_epoch} - Lightning logged val_loss (ResultMetricCollection) key not found.")
                        val_loss_metric = None 
                
                if val_loss_metric and hasattr(val_loss_metric, 'value'):
                    print(f"DEBUG: Epoch {self.current_epoch} - Lightning internal val_loss value: {val_loss_metric.value.item():.4f}")
                    if val_loss_metric.meta.is_mean_reduction and hasattr(val_loss_metric, 'cumulated_batch_size'):
                        print(f"DEBUG: Epoch {self.current_epoch} - Lightning internal val_loss cumulated_batch_size: {val_loss_metric.cumulated_batch_size.item()}")
                else:
                    print(f"DEBUG: Epoch {self.current_epoch} - Lightning internal val_loss metric not found or has no value.")
            else:
                print(f"DEBUG: Epoch {self.current_epoch} - Lightning internal val_loss key '{lightning_logged_val_loss_key}' not found in results.")
        # debug end
        self.log('val_loss',mean_val_loss, prog_bar=True)
        self.log('val_acc',mean_val_acc, prog_bar=True)
        self.log('val_iou',mean_val_iou, prog_bar=True)
        self.log('val_f1',mean_val_f1_score, prog_bar=True) 
        val_stats =  { 'loss':{'val':mean_val_loss}, 'accuracy':{'val':mean_val_acc},
                       'iou':{'val':mean_val_iou}, 'f1':{'val':mean_val_f1_score},
                       'confusion-matrix':{'val':mean_val_conf_mat},
                    }
        if self.classifier:
            mean_val_domain_loss = torch.stack([x['log']['domain_loss']['val'].float().mean() for x in val_step_outputs]).cpu().mean().item()
            mean_val_domain_acc = torch.stack([x['log']['domain_accuracy']['val'].float().mean() for x in val_step_outputs]).cpu().mean().item()
            self.custom_logs['val_domain_loss'].append(mean_val_domain_loss)
            self.custom_logs['val_domain_acc'].append(mean_val_domain_acc)
            self.log('val_domain_loss',mean_val_domain_loss, prog_bar=True)
            self.log('val_domain_acc',mean_val_domain_acc, prog_bar=True)
            domain_stats = { 'domain_loss':{'val':mean_val_domain_loss},
                             'domain_accuracy':{'val':mean_val_domain_acc}
                            }
            val_stats.update(domain_stats)

        mean_val_total_loss = torch.stack([x['loss'].float().mean() for x in val_step_outputs]).cpu().mean().item()

        return {'loss':mean_val_total_loss, 'log':val_stats, 'progress_bar':val_stats}

########## development

    def test_step(self, batch, batch_idx):
        # pdb.set_trace()
        inputs, labels, paths = batch # batch['slum_prediction']
        labels = labels.to(dtype=inputs.dtype)
        batch_size_1 = len(paths)
        outputs_ = self.model(inputs)
        outputs = outputs_[:batch_size_1]
        labels = labels.view(outputs.shape)
        pbar = self._calc_metrics((labels>0.5).long(), outputs)
        test_stats =  { 'accuracy':{'test':pbar['accuracy']},
                       'f1-score':{'test':pbar['f1-score']}, 'iou':{'test':pbar['iou']}, 
                       'confusion-matrix':{'test':pbar['confusion-matrix']},
                    }
        return {'log':test_stats}

    def test_step_end(self, test_step_outputs):
        '''agreggate across gpus (data parallel mode)'''
        if isinstance(test_step_outputs, list): 
            test_acc = torch.tensor([x['log']['accuracy']['test'].detach() for x in test_step_outputs]).mean()
            test_iou = torch.tensor([x['log']['iou']['test'].detach() for x in test_step_outputs]).mean()
            test_f1_score = torch.tensor([x['log']['f1-score']['test'].detach() for x in test_step_outputs]).mean()
            test_conf_mat = torch.stack([x['log']['confusion-matrix']['test'].detach() for x in test_step_outputs]).sum(0)
            test_stats =  { 'accuracy':{'test':test_acc},
                            'iou':{'test':test_iou}, 'f1':{'test':test_f1_score},
                            'confusion-matrix':{'test':test_conf_mat},
                        }       
            return {'log':test_stats}
        else:
            return test_step_outputs

    def test_epoch_end(self, test_step_outputs):
        '''agreggate across batches'''
        mean_test_conf_mat = torch.stack([x['log']['confusion-matrix']['test'] for x in test_step_outputs]).sum(0).clone().cpu().detach().numpy().tolist()
        macro_test_acc = (mean_test_conf_mat[0][0] + mean_test_conf_mat[1][1]) / (mean_test_conf_mat[0][0] + mean_test_conf_mat[0][1] + mean_test_conf_mat[1][0] + mean_test_conf_mat[1][1])
        macro_test_precision =  mean_test_conf_mat[0][0]/(mean_test_conf_mat[0][0]+mean_test_conf_mat[1][0])
        macro_test_recall = mean_test_conf_mat[0][0]/(mean_test_conf_mat[0][0]+mean_test_conf_mat[0][1])
        macro_test_f1_score = 2 * (macro_test_precision*macro_test_recall)/(macro_test_precision+macro_test_recall)
        macro_test_iou = (macro_test_precision * macro_test_recall) / (macro_test_precision + macro_test_recall - (macro_test_precision*macro_test_recall))
        self.log('test_macro_acc',macro_test_acc, prog_bar=False)
        self.log('test_macro_precision',macro_test_precision, prog_bar=False)
        self.log('test_macro_recall', macro_test_recall, prog_bar=False) 
        self.log('test_macro_f1', macro_test_f1_score, prog_bar=False) 
        self.log('test_macro_iou', macro_test_iou, prog_bar=False) 
        self.log('tp', float(mean_test_conf_mat[0][0]), prog_bar=False) 
        self.log('fn', float(mean_test_conf_mat[0][1]), prog_bar=False)  
        self.log('fp', float(mean_test_conf_mat[1][0]), prog_bar=False) 
        self.log('tn', float(mean_test_conf_mat[1][1]), prog_bar=False)
        test_stats =  {'accuracy':{'test':macro_test_acc},
                       'iou':{'test':macro_test_iou}, 
                       'f1':{'test':macro_test_f1_score},
                       'confusion-matrix':{'test':mean_test_conf_mat},
                       'precision':{'test':macro_test_precision},
                       'recall':{'test':macro_test_recall},
                    }
        return {'log':test_stats}

########### development

    @property
    def defaults(self):
        print("\nDefault model hyper-parameters:")
        print(MODEL_PARAMS)
        print("\nDefault training hyper-parameters:")
        print(TRAINING_PARAMS)
        print("\nList of available criteria (loss functions):")
        print(list(self.SUPPORTED_LOSSES.keys()))
        print("\nList of available optimisers:")
        print(list(self.SUPPORTED_OPTIMIZERS.keys()))

    @staticmethod
    def flatten_results_dict(results):
        """
        Expand a hierarchical dict of scalars into a flat dict of scalars.
        If results[k1][k2][k3] = v, the returned dict will have the entry
        {"k1/k2/k3": v}.
        Args:
            results (dict):
        """
        r = {}
        for k, v in results.items():
            if isinstance(v, dict):
                v = ModelTrainingWrapper.flatten_results_dict(v)
                for kk, vv in v.items():
                    r[k + "/" + kk] = vv
            else:
                r[k] = v
        return r

    def get_hparams(self):
        params = list(TRAINING_PARAMS.keys()) + list(MODEL_PARAMS.keys())
        pdict = {}
        for param in params:
            try:
                pdict[param] = getattr(self, param)
            except AttributeError:
                pdict[param] = getattr(self.model, param)
        setattr(self, 'hparams', pdict)

class StatsSavingCallback(Callback):
    '''Save training curves and validation set evaluation metrics every k number of epochs.
    At the end of training it will also plot the train-val.curves.
    Args:
        save_every:         int, the saving interval (in epochs)'''

    def __init__(self, save_every=10, save_hyperparameters=False):
        super(StatsSavingCallback, self).__init__()
        self.save_every = save_every
        self.save_hyperparameters = save_hyperparameters

    def _save_logs(self, model, save_path):
        filename = os.path.join(save_path,"train_val_curves.csv")
        logs_to_save = model.custom_logs # use original logs
        #debug use
        print(f"DEBUG: _save_logs called at epoch {model.current_epoch}.")
        print(f"DEBUG: _save_logs - Target filename: {filename}") # 增加调试输出
        print(f"DEBUG: _save_logs - Initial lengths: train_loss={len(logs_to_save['train_loss'])}, val_loss={len(logs_to_save['val_loss'])}")
        data_for_df = {}
        train_epochs_data = logs_to_save['train_loss'][1:] if len(logs_to_save['train_loss']) > 0 and logs_to_save['train_loss'][0] == 0.0 else logs_to_save['train_loss']
        val_epochs_data = logs_to_save['val_loss']
        if len(val_epochs_data) > len(train_epochs_data):
            val_offset_to_match_train = len(val_epochs_data) - len(train_epochs_data)
            print(f"DEBUG: _save_logs - Adjusting val_loss length: val_offset_to_match_train={val_offset_to_match_train}")
            val_epochs_data = val_epochs_data[val_offset_to_match_train:]
        elif len(train_epochs_data) > len(val_epochs_data):
            train_offset_to_match_val = len(train_epochs_data) - len(val_epochs_data)
            print(f"DEBUG: _save_logs - Adjusting train_loss length: train_offset_to_match_val={train_offset_to_match_val}")
            train_epochs_data = train_epochs_data[train_offset_to_match_val:]
        
        data_for_df['train_loss'] = train_epochs_data
        data_for_df['val_loss'] = val_epochs_data

        for k, v in logs_to_save.items():
            if k not in ['train_loss', 'val_loss']: 
                if k.startswith('train_'):
                    data_for_df[k] = v[1:] if len(v) > 0 and v[0] == 0.0 else v
                elif k.startswith('val_'):
                    if len(v) > len(train_epochs_data):
                        data_for_df[k] = v[len(v) - len(train_epochs_data):]
                    else:
                        data_for_df[k] = v
                else: 
                    data_for_df[k] = v

        min_len = min(len(v) for v in data_for_df.values())
        print(f"DEBUG: _save_logs - Minimum length for DataFrame: {min_len}")
        final_data_for_df = {k: v[:min_len] for k, v in data_for_df.items()}
        print(f"DEBUG: _save_logs - Final lengths for DataFrame: {[len(v) for v in final_data_for_df.values()]}")

        try:
            df = pd.DataFrame.from_dict(final_data_for_df) 
            print(f"DEBUG: _save_logs - DataFrame shape before saving: {df.shape}")
            df.to_csv(filename)
            print(f"DEBUG_SAVE: Epoch {model.current_epoch} - Saved train_val_curves.csv with shape {df.shape}")

            #debugging ===
            try:
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    print(f"DEBUG: _save_logs - train_val_curves.csv exists. Size: {file_size} bytes.")
                    re_read_df = pd.read_csv(filename, header=0, index_col=0)
                    print(f"DEBUG: _save_logs - Successfully re-read CSV. Shape: {re_read_df.shape}")
                    print(f"DEBUG: _save_logs - Last rows of re-read CSV:\n{re_read_df.tail()}")
                else:
                    print(f"ERROR: _save_logs - train_val_curves.csv DOES NOT EXIST after writing attempt.")
            except Exception as e:
                print(f"ERROR: _save_logs - Failed to re-read CSV after writing: {e}")

        except Exception as Error:
            # === advanced error traceback ===
            print(f"ERROR: _save_logs - Problem saving logs at epoch {model.current_epoch}! Error: {Error}")
            import traceback
            traceback.print_exc()
            # pdb.set_trace() 
            print("Problem saving logs. Logs data:", {k:len(v) for k,v in logs_to_save.items()}) 
            print("Processed logs data for DataFrame:", {k:len(v) for k,v in final_data_for_df.items()}) 

    def _save_hyperparams(self, model, save_path):
        if self.save_hyperparameters:
            filename = os.path.join(save_path,"hyperparameters.json")
            try:
                with open(filename, "w") as fout:
                    json.dump(model.hparams, fout, indent=4)
                print(f"DEBUG_SAVE: Saved hyperparameters.json at {filename}. Size: {os.path.getsize(filename)} bytes.")
            except Exception as e:
                print(f"ERROR: Failed to save hyperparameters.json: {e}")


    def _save_training_curves(self, trainer, metric="val_f1", filename=f"training_curves-{str(time.time()).split('.')[0]}.png", LOG_LOSS_SCALE=True):
        csv_filepath = os.path.join(trainer.log_dir, "train_val_curves.csv")
        full_output_path = os.path.join(trainer.log_dir, filename)
        
        print(f"DEBUG: _save_training_curves called. Attempting to load from: {csv_filepath}")
        if not os.path.exists(csv_filepath):
            print(f"ERROR: _save_training_curves: CSV file not found at {csv_filepath}. Cannot plot curves.")
            return

        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        print(f"DEBUG: _save_training_curves: Loaded DataFrame shape: {df.shape}")

        if df.empty:
            print(f"WARNING: _save_training_curves: DataFrame is empty, skipping plot.")
            return

        # Ensure columns exist before plotting
        required_cols = ['train_loss', 'val_loss', 'val_f1', 'val_iou']
        if not all(col in df.columns for col in required_cols):
            print(f"ERROR: _save_training_curves: Missing required columns for plotting. Found: {df.columns.tolist()}")
            return


        tstamp = str(datetime.datetime.now()).split('.')[0]
        plt.figure(figsize=(14,10))
        plt.subplot(2,1,1)
        plt.plot(df['train_loss'], 'black', label='train_loss')
        plt.plot(df['val_loss'],'blue',label='val_loss')
        plt.title("Loss curves")
        plt.xlabel("Epoch number")
        
        if not df['val_loss'].empty:
            min_val_loss_epoch = df['val_loss'].argmin()
            min_val_loss = df['val_loss'].min()
            plt.axvline(x=min_val_loss_epoch, color='red', label=f"val loss min={min_val_loss:0.3f}")
            plt.axhline(y=min_val_loss, color='red')
        else:
            print(f"WARNING: _save_training_curves: val_loss is empty, skipping min loss line.")

        plt.legend(loc='upper right')
        plt.grid(True)
        if LOG_LOSS_SCALE:
            plt.yscale('log')
        plt.subplot(2,1,2)
        plt.plot(df['val_f1'], 'blue', label='val_f1')
        plt.plot(df['val_iou'], 'green', label='val_iou')
        plt.xlabel("Epoch number")
        plt.title("Validation metrics")
        plt.grid(True)
        
        if not df[metric].empty:
            max_metric_epoch = df[metric].argmax()
            max_metric_value = df[metric].max()
            plt.axvline(x=max_metric_epoch,color='red', label=f"{metric} max={max_metric_value:0.3f}")
            plt.axhline(y=max_metric_value, color='red')
        else:
            print(f"WARNING: _save_training_curves: {metric} is empty, skipping max metric line.")

        plt.legend(loc='lower right')
        plt.suptitle("Training curves for run completed on "+tstamp, fontweight='bold')
        plt.tight_layout()
        plt.savefig(full_output_path)
        plt.close()
        print(f"DEBUG_SAVE: Saved training_curves plot to {full_output_path}")


    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module
        
        print(f"DEBUG: on_train_epoch_end called for epoch {model.current_epoch}.")
        print(f"DEBUG: self.save_every={self.save_every}")
        print(f"DEBUG: Trainer.log_dir at on_train_epoch_end: {trainer.log_dir}")


        # Ensure logs are saved if current epoch is a multiple of save_every or it's the first epoch (after initial 0.0)
        # Note: model.current_epoch is 0-indexed, so for epoch 1, current_epoch is 0
        # If your intention is to save at the *end* of epoch 1, 2, 3...
        # and not after every batch if log_every_n_steps=1 (which save_every=1 implies)
        # then the logic seems fine.
        if (model.current_epoch % self.save_every == 0) and (model.current_epoch > 0):
            print(f"DEBUG: Condition met for _save_logs (train) at epoch {model.current_epoch}.")
            self._save_logs(model, trainer.log_dir)
        elif model.current_epoch == 0 and self.save_every == 1: # Special case for epoch 0, if save_every is 1
            print(f"DEBUG: Condition met for _save_logs (train, Epoch 0 special save) at epoch {model.current_epoch}.")
            self._save_logs(model, trainer.log_dir)
        else:
            print(f"DEBUG: Condition not met for _save_logs (train) at epoch {model.current_epoch}. Skipping save.")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module) # Call parent method
        model = pl_module # maintain the same with original code
        print(f"DEBUG: on_validation_epoch_end called for epoch {model.current_epoch}.")
        print(f"DEBUG: self.save_every={self.save_every}")
        print(f"DEBUG: Trainer.log_dir at on_validation_epoch_end: {trainer.log_dir}")

        if (model.current_epoch % self.save_every == 0) and (model.current_epoch > 0):
            print(f"DEBUG: Condition met for _save_logs (val) at epoch {model.current_epoch}.")
            self._save_logs(model, trainer.log_dir)
        elif model.current_epoch == 0 and self.save_every == 1: # Special case for epoch 0, if save_every is 1
            print(f"DEBUG: Condition met for _save_logs (val, Epoch 0 special save) at epoch {model.current_epoch}.")
            self._save_logs(model, trainer.log_dir)
        else:
            print(f"DEBUG: Condition not met for _save_logs (val) at epoch {model.current_epoch}. Skipping save.")

    def on_train_end(self, trainer, model):
        # === Debugging ===
        print(f"DEBUG: on_train_end called.")
        print(f"DEBUG: Final epoch number: {model.current_epoch}.")
        print(f"DEBUG: Trainer.log_dir at on_train_end: {trainer.log_dir}") # 增加调试输出
        # ====================
        
        # Save logs one last time to ensure all data is captured
        self._save_logs(model, trainer.log_dir)
        self._save_training_curves(trainer, metric="val_f1")

class InspectPredictionsCallback(Callback):
    '''Save triplets of (images, labels, predictions) for the validation set 
    every a defined number of epochs.
    Args:
        save_every:         int, the saving interval (in epochs)
        norm_file:          str, lpath to the normalization file used 
        n_batches:          int, how many batches from the validation set to save [default:5]
    Images are saved in a the folder \'PredictionSamples\' in the results dir.
    '''
    def __init__(self, save_every=3, norm_file=None, dataset=None, n_batches=5):
        super(InspectPredictionsCallback, self).__init__()
        self.save_every = save_every
        self.dataset = dataset
        self.norm_file = norm_file
        if norm_file is not None:
            with open(norm_file, 'r') as json_file:
                data = json.load(json_file)
            self.mean = np.array(data["mean"]) # Ensure numpy array for broadcasting
            self.std = np.array(data["std"])   # Ensure numpy array for broadcasting
            print(f"DEBUG: InspectPredictionsCallback: Loaded normalization mean={self.mean}, std={self.std}")
        else:
            print("Normalization File not Provided. Images will be un-normalized.")
            self.mean = np.array([0.])
            self.std = np.array([1.])
        if n_batches == -1:
            self.n_batches = set(range(10000))
        elif n_batches == 0:
            self.n_batches = set([])
        else:
            self.n_batches = set(range(n_batches))
        print(f"DEBUG: InspectPredictionsCallback initialized. Saving {len(self.n_batches)} batches every {self.save_every} epochs.")

    def get_create_base_path(self, trainer):
        self.base_path = os.path.join(trainer.log_dir, "PredictionSamples")
        os.makedirs(self.base_path, exist_ok=True)
        print(f"DEBUG: InspectPredictionsCallback: Base path for predictions: {self.base_path}. Created: {os.path.exists(self.base_path)}")

    def on_train_start(self, trainer, model):
        self.get_create_base_path(trainer)
        print(f"DEBUG: InspectPredictionsCallback.on_train_start: Trainer.log_dir: {trainer.log_dir}")

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
        # This hook is called after every validation batch.
        if (model.current_epoch % self.save_every  == 0) and (model.current_epoch > 0) and (batch_idx in self.n_batches):
            print(f"DEBUG: InspectPredictionsCallback.on_validation_batch_end: Saving batch {batch_idx} at epoch {model.current_epoch}.")
            if not hasattr(self, 'base_path'):
                self.get_create_base_path(trainer)            
            self.current_epoch = format(model.current_epoch, '05d')
            # Robustly unpack
            
            if isinstance(batch, dict) and 'slum_prediction' in batch:
                # Case 1: Batch comes from a CombinedLoader.
                imgs, dino_features, lbls, names = batch['slum_prediction']
            elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                # Case 2: Batch comes from a standard DataLoader.
                imgs, dino_features, lbls, names = batch
            else:
                # If the format is unexpected, print a warning and skip saving this batch.
                print(f"WARNING: InspectPredictionsCallback received an unsupported batch format: {type(batch)}. Skipping image saving for this batch.")
                return

            # Move inputs to the model's device (batch arrives on CPU in the callback before DP scattering).
            _dev = model.device
            imgs = imgs.to(_dev)
            dino_features = dino_features.to(_dev)
            y_hat_logits, _ = model(imgs, dino_features) if model.classifier else (model(imgs, dino_features), None)
            y_hat = torch.sigmoid(y_hat_logits) > model.hparams.threshold # Use sigmoid for logits
            
            y_hat = torch.squeeze(y_hat,dim=1).detach().cpu().numpy() * 255
            lbls_ = torch.squeeze(lbls, dim=1).detach().cpu().numpy() * 255
            
            # Create RGB visualizations
            out_y_hat = np.stack([y_hat, np.zeros_like(y_hat), np.zeros_like(y_hat)], axis=3) # Predictions in Red
            out_lbls = np.stack([np.zeros_like(lbls_), lbls_, np.zeros_like(lbls_)], axis=3)  # Labels in Green
            
            # Un-normalize the image for visualization
            out_imgs = np.transpose(copy.deepcopy(imgs.detach().cpu().numpy()), (0,2,3,1)) 
            if out_imgs.shape[3] > 3: # Handle multi-channel images (e.g., take first 3 channels)
                out_imgs = out_imgs[:,:,:,:3]
            out_imgs = (((out_imgs * self.std[:out_imgs.shape[3]]) + self.mean[:out_imgs.shape[3]]) * 255).astype(np.uint8)
            
            combined = np.concatenate([out_imgs, out_lbls, out_y_hat], axis=2)
            img_iterator = [combined[i].astype(np.uint8) for i in range(combined.shape[0])]
            self.save_batch_images(img_iterator, names)

    def save_batch_images(self, y_hat_iterator, names_iterator, format:str='png'):
        for y_, name in zip(y_hat_iterator, names_iterator):
            output_filepath = os.path.join(self.base_path, 'epoch_'+self.current_epoch)+'_'+name
            try:
                imageio.imwrite(uri=output_filepath, 
                                im=imgmat2img(y_), format=format)
                print(f"DEBUG_SAVE: Saved prediction sample to {output_filepath}")
            except Exception as e:
                print(f"ERROR: Failed to save image {output_filepath}. Error: {e}")


class PredictOnTestSetCallback(Callback):
    '''Save triplets of (images, labels, predictions) from the test set 
    every a defined number of epochs.
    Args:
        save_every:         int, the saving interval (in epochs)
        img_list:           list[path], list of paths to images for which predictions will 
                            be made and saved
    Images are saved in a the folder \'PredictionSamples\' in the results dir.
    '''

    def __init__(self, save_every=3, datapath=None, norm_file=None, batch_size=25, num_workers=4, tile_size=512):
        super(PredictOnTestSetCallback, self).__init__()
        self.save_every = save_every
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm_file = norm_file
        self.tile_size = tile_size
        print(f"DEBUG: PredictOnTestSetCallback initialized. Predicting every {self.save_every} epochs for datapath: {self.datapath}")

    # def _setup(self):
    #     self.dl = InferenceDataLoader(datapath=self.datapath, norm_file=self.norm_file, 
    #                              batch_size=self.batch_size, num_workers=self.num_workers, tile_size=self.tile_size)
    #     self.DataLoader = iter(self.dl.get_dataloader())

    def _setup(self):
        print(f"DEBUG: PredictOnTestSetCallback._setup called. Datapath: {self.datapath}")
        if self.datapath is None:
            print("WARNING: PredictOnTestSetCallback: datapath is None. Skipping DataLoader setup.")
            self.dl = None
            return
        
        # Check if the data path is valid
        if not Path(self.datapath).exists():
            print(f"ERROR: PredictOnTestSetCallback: Data path {self.datapath} does not exist. Cannot create DataLoader.")
            self.dl = None
            return

        try:
            self.dl = InferenceDataLoader(datapath=self.datapath, norm_file=self.norm_file, 
                                    batch_size=self.batch_size, num_workers=self.num_workers, tile_size=self.tile_size)
            self.DataLoader = iter(self.dl.get_dataloader())
            print(f"DEBUG: PredictOnTestSetCallback: DataLoader created successfully. First batch check...")
            try:
                # Attempt to get a batch to ensure the DataLoader is functional
                first_batch = next(iter(self.dl.get_dataloader()))
                print(f"DEBUG: PredictOnTestSetCallback: Successfully loaded first batch from inference DataLoader. Keys: {list(first_batch.keys())}")
            except StopIteration:
                print("WARNING: PredictOnTestSetCallback: Inference DataLoader is empty after initialization.")
            except Exception as e:
                print(f"ERROR: PredictOnTestSetCallback: Error loading first batch from inference DataLoader: {e}")

        except Exception as e:
            print(f"ERROR: PredictOnTestSetCallback: Failed to setup InferenceDataLoader. Error: {e}")
            self.dl = None # Set to None if setup fails

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_model_obj: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_model_obj) # Call parent method
        print(f"DEBUG: PredictOnTestSetCallback.on_validation_epoch_end called for epoch {pl_model_obj.current_epoch}.")
        print(f"DEBUG: Trainer.log_dir at PredictOnTestSetCallback: {trainer.log_dir}")


        if (pl_model_obj.current_epoch % self.save_every  == 0) and (pl_model_obj.current_epoch > 0):
            print(f"DEBUG: PredictOnTestSetCallback: Condition met for predicting full map at epoch {pl_model_obj.current_epoch}.")
            pl_model_obj.eval() # Set model to evaluation mode
            
            self.base_path = os.path.join(trainer.log_dir, "Epoch_"+format(pl_model_obj.current_epoch, '05d')+"_SlumMap")
            os.makedirs(self.base_path, exist_ok=True)
            print(f"DEBUG: PredictOnTestSetCallback: Created output directory: {self.base_path}")

            self._setup() # Setup DataLoader for inference
            if self.dl is None or self.DataLoader is None:
                print("WARNING: PredictOnTestSetCallback: DataLoader not initialized, skipping prediction.")
                pl_model_obj.train() # Set model back to train mode
                return

            preds, names = self.predict_for_dataset(pl_model_obj, pl_model_obj.current_epoch>1)
            for imgs_batch, names_batch in zip(preds, names):
                self.save_batch_images(imgs_batch, names_batch)
            
            reconstructed_path = os.path.join(trainer.log_dir, "reconstructed_map_epoch_{}.png".format(pl_model_obj.current_epoch))
            try:
                ImageTiler.reconstruct_image(self.base_path, reconstructed_path)
                print(f"DEBUG_SAVE: Reconstructed image saved to {reconstructed_path}")
            except Exception as e:
                print(f"ERROR: PredictOnTestSetCallback: Failed to reconstruct image from {self.base_path}. Error: {e}")

            pl_model_obj.train() # Set model back to train mode

    #TODO: move everything to cpu
    # @torch.no_grad()
    def predict_for_dataset(self, pl_model_obj, FLAG):
        preds = []
        names = []
        # pl_model_obj.model.eval() # Already set in on_validation_epoch_end
        
        # Reset dataloader iterator to ensure fresh start
        self.DataLoader = iter(self.dl.get_dataloader())
        
        for i, batch in enumerate(self.DataLoader):
            if i == 0:
                print(f"DEBUG: PredictOnTestSetCallback.predict_for_dataset: Processing first batch (idx 0).")
            imgs, names_batch = batch['slum_prediction']
            
            # Ensure model and input are on the same device
            imgs = imgs.to(pl_model_obj.device) 
            y_hat = pl_model_obj(imgs) > pl_model_obj.threshold
            y_hat = torch.squeeze(y_hat,dim=1).detach().cpu().numpy() * 255
            preds.append(np.stack([y_hat, np.zeros_like(y_hat), np.zeros_like(y_hat)],axis=3))
            names.append(names_batch)
            if i % 10 == 0: # Print progress
                print(f"DEBUG: PredictOnTestSetCallback.predict_for_dataset: Processed {i+1} batches.")

        # pl_model_obj.model.train() # Already set in on_validation_epoch_end
        print(f"DEBUG: PredictOnTestSetCallback.predict_for_dataset: Finished processing dataset. Total batches: {i+1 if 'i' in locals() else 0}")
        return preds, names

    def save_batch_images(self, imgs_iterator, names_iterator, format:str='png'):
        for img_i, name in zip(imgs_iterator, names_iterator):
            output_filepath = os.path.join(self.base_path, name)
            try:
                im_i = imgmat2img(img_i)
                imageio.imwrite(uri=output_filepath, 
                                im=im_i, format=format)
                print(f"DEBUG_SAVE: Saved predicted map tile to {output_filepath}")
            except Exception as e:
                print(f"ERROR: Failed to save predicted map tile {output_filepath}. Error: {e}")

def imgmat2img(imgmat, IMGTYPE='uint8'):
    '''Helper function that makes sure we can save images of different channel depths.
    It will convert single channel images to 3-channel grayscales 
    and will keep the first 3 channels of 4 channel ones. 
    Will convert float values to 8bit ints.'''
    assert isinstance(imgmat, np.ndarray), "Image matrix not a numpy array"
    img_in = copy.deepcopy(imgmat)
    if len(img_in.shape) > 3:
        return img_in[:,:,:3].astype(IMGTYPE)
    if len(img_in.shape) < 3:
        img_in = np.atleast_3d(img_in)
    if img_in.shape[-1] == 1:
        return np.repeat(img_in, repeats=3, axis=2).astype(IMGTYPE)
    else:
        return img_in.astype(IMGTYPE)

class ConfigSavingCallback(Callback):
    '''Save the training configuration (at the begining of training)
    and the training normalization file provided (for use in inference and evaluation)
    Args:
        config:         str, the training configuration yml file (e.g. base_savio.yml)'''

    def __init__(self, used_config, config_filepath=None):
        super(ConfigSavingCallback, self).__init__()
        self.used_config = used_config
        self.config_path = config_filepath
        print(f"DEBUG: ConfigSavingCallback initialized. Config file path provided: {self.config_path}")

    def _save_config(self, save_path):
        destination = os.path.join(save_path,"config.yml")
        print(f"DEBUG: ConfigSavingCallback: Attempting to save config to {destination}")
        if self.config_path is not None and os.path.exists(self.config_path):
            try:
                shutil.copyfile(self.config_path, destination)
                print(f"DEBUG_SAVE: Copied config file from {self.config_path} to {destination}. Size: {os.path.getsize(destination)} bytes.")
            except Exception as Error:
                print(f"ERROR: ConfigSavingCallback: Error copying config file {self.config_path} to {destination}. Error: {Error}")
                import traceback
                traceback.print_exc()
        else:
            try:
                with open(destination, "w") as fout:
                    yml = ruamel.yaml.YAML()
                    yml.indent(mapping=4, sequence=4, offset=2)
                    yml.preserve_quotes = True
                    yml.dump(self.used_config, fout)
                print(f"DEBUG_SAVE: Dumped config to {destination}. Size: {os.path.getsize(destination)} bytes.")
            except Exception as Error:
                print("ERROR: ConfigSavingCallback: Error while saving used config file. Continuing without saving...\n", Error)
                import traceback
                traceback.print_exc()

    def _save_normalization_stats(self, save_path):
        destination = os.path.join(save_path,"normalization.json")
        norm_file_source = self.used_config['paths'].get('normalization_file')
        print(f"DEBUG: ConfigSavingCallback: Attempting to save normalization stats to {destination} from {norm_file_source}")
        if norm_file_source and os.path.exists(norm_file_source):
            try:
                shutil.copyfile(norm_file_source, destination)
                print(f"DEBUG_SAVE: Copied normalization file from {norm_file_source} to {destination}. Size: {os.path.getsize(destination)} bytes.")
            except Exception as Error:
                print("ERROR: ConfigSavingCallback: Error while saving used normalization file. Continuing without saving...\n", Error)
                import traceback
                traceback.print_exc()
        else:
            print(f"WARNING: ConfigSavingCallback: Normalization file source {norm_file_source} not found or not specified. Skipping normalization stats save.")

    def _save_performance_csv_line(self, trainer):
        csv_filepath = os.path.join(trainer.log_dir, "train_val_curves.csv")
        out_filename = os.path.join(trainer.log_dir, "training_register.csv")
        print(f"DEBUG: ConfigSavingCallback: Attempting to create training_register.csv from {csv_filepath}")
        if not os.path.exists(csv_filepath):
            print(f"ERROR: ConfigSavingCallback: train_val_curves.csv not found at {csv_filepath}. Cannot create training_register.csv.")
            return

        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        
        if df.empty:
            print(f"WARNING: ConfigSavingCallback: train_val_curves.csv is empty. Skipping training_register.csv creation.")
            return

        if self.used_config['run_type']['pretrained_model'] is not None:
            model_type = self.used_config['run_type']['pretrained_model']
        else:
            model_type = 'unet'
        
        # Safely access DataFrame columns, providing defaults if not present or empty
        best_f1_epoch = df['val_f1'].argmax() if 'val_f1' in df.columns and not df['val_f1'].empty else -1
        best_f1_value = df['val_f1'].max() if 'val_f1' in df.columns and not df['val_f1'].empty else np.nan
        
        best_iou_epoch = df['val_iou'].argmax() if 'val_iou' in df.columns and not df['val_iou'].empty else -1
        best_iou_value = df['val_iou'].max() if 'val_iou' in df.columns and not df['val_iou'].empty else np.nan

        min_val_loss_epoch = df['val_loss'].argmin() if 'val_loss' in df.columns and not df['val_loss'].empty else -1
        min_val_loss_value = df['val_loss'].min() if 'val_loss' in df.columns and not df['val_loss'].empty else np.nan

        best_conf_matrix = df.loc[best_f1_epoch,'val_conf_mat'] if 'val_conf_mat' in df.columns and not df['val_conf_mat'].empty and best_f1_epoch != -1 else 'N/A'


        description_list = [
            ('version', str(Path(trainer.log_dir).name).split('_')[-1]),
            ('model_type', model_type),
            ('loss_function', self.used_config['training_parameters']['criterion']),
            ('optimizer', self.used_config['training_parameters']['optimizer']),
            ('num_epochs', self.used_config['training_parameters']['num_epochs']),
            ('effective_batch_size', self.used_config['training_parameters']['batch_size']*self.used_config['training_parameters']['grad_acc_steps']),
            ('training_path', self.used_config['paths']['training_csv']),
            ('normalization_file', self.used_config['paths']['normalization_file']),
            ('min_val_loss', min_val_loss_value),
            ('min_val_loss_epoch', min_val_loss_epoch),
            ('max_val_f1', best_f1_value),
            ('max_val_f1_epoch', best_f1_epoch),
            ('max_val_iou', best_iou_value),
            ('best_conf_matrix', best_conf_matrix)
        ]
        dfd = pd.DataFrame.from_records(description_list).T
        dfd.columns = dfd.iloc[0,:]
        dfd = dfd.drop(0,axis=0).reset_index(drop=True) # drop=True to avoid adding new index column
        try:
            dfd.to_csv(out_filename, index=False) # index=False to prevent writing index column
            print(f"DEBUG_SAVE: Saved training_register.csv to {out_filename}. Size: {os.path.getsize(out_filename)} bytes.")
        except Exception as e:
            print(f"ERROR: ConfigSavingCallback: Failed to save training_register.csv. Error: {e}")
            import traceback
            traceback.print_exc()

    def on_train_start(self, trainer, model):
        super().on_train_start(trainer, model) # Call parent method
        print(f"DEBUG: ConfigSavingCallback.on_train_start called. Trainer.log_dir: {trainer.log_dir}")
        self._save_config(save_path=trainer.log_dir)
        self._save_normalization_stats(save_path=trainer.log_dir)

    def on_train_end(self, trainer, model):
        super().on_train_end(trainer, model) # Call parent method
        print(f"DEBUG: ConfigSavingCallback.on_train_end called. Trainer.log_dir: {trainer.log_dir}")
        self._save_performance_csv_line(trainer)
        #delete unused hparams.yml file
        hparams_file_path = os.path.join(trainer.log_dir, 'hparams.yaml')
        # safe deletion
        print(f"DEBUG: Checking for hparams.yaml at: {hparams_file_path}")
        if os.path.exists(hparams_file_path):
            try:
                os.remove(hparams_file_path)
                print(f"DEBUG: Successfully removed hparams.yaml at: {hparams_file_path}")
            except Exception as e:
                print(f"ERROR: Failed to remove hparams.yaml at {hparams_file_path}. Error: {e}")
        else:
            print(f"DEBUG: hparams.yaml NOT found at {hparams_file_path}. Skipping removal.")

class CheckBatchGradient(pl.Callback):
    ''' Pass a batch through the model to check if there are batch mixing errors in your code.
    The idea is that passing a batch through the model and taking the derivative w.r.t. one input
    should have no effect on the others. If not then the model mixes the batch (probably sums allong 
    the wrong dimension somewhere?).
    Coppied from: https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch
    CANNOT BE USED WITH BATCHNORM - BN introduces dependencies across the batch due to the normalization with batch statistcs
    # use the callback like this:
        >>> model = LitClassifier()
        >>> trainer = pl.Trainer(gpus=1, callbacks=[CheckBatchGradient()])
        >>> trainer.fit(model)
    '''

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
        n = 0
        if 'slum_prediction' in batch:
            inputs, lbls, names = batch['slum_prediction'] # Ensure correct batch unpacking
            print(f"DEBUG: CheckBatchGradient: Successfully unpacked 'slum_prediction' from batch.")
        else:
            print(f"WARNING: CheckBatchGradient: 'slum_prediction' key not found in batch at batch_idx {batch_idx}. Skipping gradient check.")
            return

        example_input = copy.deepcopy(inputs)
        example_input.requires_grad_(True)
        print(f"DEBUG: CheckBatchGradient: example_input.requires_grad={example_input.requires_grad}")

        model.zero_grad()
        output = model(example_input)
        
        # Adjust for domain classifier output
        if isinstance(output, tuple) and len(output) == 2:
            segmentation_output = output[0]
            print(f"DEBUG: CheckBatchGradient: Model output is a tuple (segmentation, domain_classifier). Using segmentation output.")
        else:
            segmentation_output = output
            print(f"DEBUG: CheckBatchGradient: Model output is single (segmentation).")

        # Ensure n is within bounds for segmentation_output
        if segmentation_output.size(0) <= n:
            print(f"WARNING: CheckBatchGradient: Batch size ({segmentation_output.size(0)}) is too small for n={n}. Skipping gradient check.")
            return

        pseudo_loss = segmentation_output[n].abs().sum()
        pseudo_loss.backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad is not None:
            if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
                print("SHAPE:", example_input.shape)
                print("SUM of GRADS:",example_input.grad[zero_grad_inds].abs().sum().item())
                raise RuntimeError("Your model mixes data across the batch dimension!")
            else:
                print("GRADIENT MIXING TEST PASSED! Training can proceed...")
        else:
            print("DEBUG: CheckBatchGradient: example_input.grad is None. No gradient check performed (likely no gradients needed for this input).")
class InputMonitor(pl.Callback):
    '''add histograms of input and target to identify possible normalization issues'''

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            if 'slum_prediction' in batch:
                x, y, _ = batch['slum_prediction'] # Unpack inputs and targets
                logger = trainer.logger
                if logger:
                    # If multiple loggers, iterate. Otherwise use directly.
                    loggers = trainer.loggers if isinstance(trainer.loggers, list) else [trainer.loggers]
                    for lg in loggers:
                        if hasattr(lg.experiment, 'add_histogram'):
                            lg.experiment.add_histogram("input", x, global_step=trainer.global_step)
                            lg.experiment.add_histogram("target", y, global_step=trainer.global_step)
                            print(f"DEBUG: InputMonitor: Logged input and target histograms for batch {batch_idx} using {type(lg).__name__}.")
                        else:
                            print(f"DEBUG: InputMonitor: Logger {type(lg).__name__} does not support add_histogram.")
                else:
                    print(f"DEBUG: InputMonitor: No logger found for trainer.")
            else:
                print(f"WARNING: InputMonitor: 'slum_prediction' key not found in batch at batch_idx {batch_idx}. Skipping histogram logging.")


# You can retrieve the checkpoint after training by calling

# >>> checkpoint_callback = ModelCheckpoint(
#      monitor='val/loss',
#      mode='min',
#      dirpath='my/path/',
#      filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
#      auto_insert_metric_name=False
#  )
# if you monitor a metric, then change the mode to max, i.e.
# >>> checkpoint_callback = ModelCheckpoint(
#      monitor='val/f1',
#      mode='max',
#      dirpath='my/path/',
#      filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
#      auto_insert_metric_name=False
#  )
# trainer = Trainer(callbacks=[checkpoint_callback])
# trainer.fit(model)
# checkpoint_callback.best_model_path

# You can disable checkpointing by passing

# trainer = Trainer(checkpoint_callback=False)


# saving with accelerators (e.g. ddp)
# alwasy use the save_checkpoint method

    # trainer = Trainer(accelerator="ddp")
    # model = MyLightningModule(hparams)
    # trainer.fit(model)
    # # Saves only on the main process
    # trainer.save_checkpoint("example.ckpt")

    # model = UnetTrainer.load_from_checkpoint('lightning_logs/version_54/checkpoints/epoch=34-step=2029.ckpt', \
    #                                         map_location={'cuda:0':'cuda:0'})

# class StochasticWeightAveraging(Callback):
