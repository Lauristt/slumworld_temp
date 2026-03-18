
"""Script for training a single model.
Basic usage:\n
        >>> python train.py -c ../config/base_savio.yml
Supported parameters:
     SUPPORTED_LOSSES= {'BinaryCrossEntropyWithLogits':BCEWithLogitsLoss(),
                       'BinarySoftF1':BinarySoftF1Loss(),
                       'DiceLoss':DiceLoss(mode='binary'), 
                       'BinaryFocalLoss':BinaryFocalLoss(), 
                       'BinaryLovaszLoss':BinaryLovaszLoss()} 
    SUPPORTED_OPTIMIZERS = { 'Adam':optim.Adam,
                             'SGD':optim.SGD
                        }
    SUPPORTED_SCHEDULERS = { 'CosineWarmRestarts':{'scheduler':optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                                   'params':{'T_0':300,'T_mult':1.25}},
                             'Plataeu':{'scheduler':optim.lr_scheduler.ReduceLROnPlateau,
                                        'params':{'mode': 'min', 'factor': 0.33, 'patience': 5, 'threshold':1e-4}},
                             'Exponential':{'scheduler':optim.lr_scheduler.ExponentialLR, 
                                            'params':{'gamma':0.95}},
                             'OneCycle':{'scheduler':optim.lr_scheduler.OneCycleLR, 
                                            'params':{'max_lr':None}},
Supported validation metrics to use for early stopping:
    - val_loss              validation set loss
    - val_f1_score          validation set f1 score
    - val_iou               validation set intersection over union
    - val_acc               validation set accuracy

DataParallel mode (dp)
Precision (32)
"""
from io import DEFAULT_BUFFER_SIZE
import json
import sys
import os
import subprocess
import torch

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__package__ = os.path.dirname(sys.path[0])
import logging
import argparse
import math
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging 
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger,TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

import ruamel.yaml
import copy
import time
import pdb

try:
    from slumworldML.src.model import MODELS_REGISTRY
    from slumworldML.src.SatelliteDataset import SatelliteDataModule, TestingDataLoader
    from slumworldML.src.trainer import (StatsSavingCallback, ModelTrainingWrapper, InspectPredictionsCallback,
                        PredictOnTestSetCallback, ConfigSavingCallback, CheckBatchGradient)
except Exception as Error:
    try:
        from src.model import MODELS_REGISTRY
        from src.SatelliteDataset import SatelliteDataModule, TestingDataLoader
        from src.trainer import (StatsSavingCallback, ModelTrainingWrapper, InspectPredictionsCallback,
                                PredictOnTestSetCallback, ConfigSavingCallback, CheckBatchGradient)
    except Exception as Error2:
        try:
            from .src.model import MODELS_REGISTRY
            from .src.SatelliteDataset import SatelliteDataModule
            from .src.trainer import (StatsSavingCallback, ModelTrainingWrapper, InspectPredictionsCallback, 
                                    PredictOnTestSetCallback, ConfigSavingCallback, CheckBatchGradient)
        except Exception as Error3:
            from ..src.model import MODELS_REGISTRY
            from ..src.SatelliteDataset import SatelliteDataModule, TestingDataLoader
            from ..src.trainer import (StatsSavingCallback, ModelTrainingWrapper, InspectPredictionsCallback, 
                                    PredictOnTestSetCallback, ConfigSavingCallback, CheckBatchGradient)

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")

def load_from_config(cfg_file):
    try:
        yml = ruamel.yaml.YAML()
        yml.indent(mapping=4, sequence=4, offset=2)
        yml.preserve_quotes = True
        with open(cfg_file, 'r') as cfg:
            yaml_txt = cfg.read()
        conf = yml.load(yaml_txt)
    except Exception as Error:
        logger.error("Config loading error! Aborting.\n", exc_info=1)
        sys.exit(1)
    return conf

def update_config(conf, cl_vars):
    newconf = copy.deepcopy(conf)
    for k,v in cl_vars.items():
        if v is not None and type(v) is not bool:
            for ck in conf.keys():
                if k in conf[ck].keys():
                    newconf[ck][k] = v
    return newconf

def lr_finder(model, dblock, conf, run_name, resume_from_checkpoint=None):
    tuner = Trainer(auto_lr_find=True, 
                    gpus=conf['compute_parameters']['gpus'],
                    auto_select_gpus=conf['compute_parameters']['auto_select_gpus'],
                    resume_from_checkpoint=resume_from_checkpoint,
                    accumulate_grad_batches=conf['training_parameters']['grad_acc_steps'])
    finder = tuner.tune(model, dblock)
    print("Suggested learning rate:", model.learning_rate)
    fig = finder['lr_find'].plot(suggest=True)
    fig.savefig(os.path.join(conf['paths']['output_dir'], 
                run_name+"-lrfinder.png"))
    sys.exit(0)

def create_run_name(conf):
    model_type = conf['run_type']['pretrained_model'] if conf['run_type']['pretrained_model'] is not None else 'unet'
    ct = str(time.time()).split('.')[0]
    sched = conf['training_parameters']['scheduler']['name'] if conf['training_parameters']['scheduler']['name'] else ''
    fold_info = 'Fold'+str(conf['run_type']['foldID']) if conf['run_type']['foldID'] >= 0 else 'SingleFold'
    RUN_NAME =  model_type+'-'+conf['training_parameters']['image_type']+'-'+\
                conf['training_parameters']['criterion']+'-'+\
                conf['training_parameters']['optimizer']+'-'+\
                sched+'-L2_'+ str(conf['training_parameters']['l2_reg'])+'-Seed_'+ \
                str(conf['training_parameters']['random_seed'])+'-'+\
                "TStamp_"+ct+'-'+\
                fold_info
    return RUN_NAME

def model_trainer(cl_vars):
    # logger = TensorBoardLogger(save_dir=tb_logs_folder, name='Classifier')
    conf = load_from_config(cl_vars['config_file'])
    if not cl_vars['from_config']:
        conf = update_config(conf, cl_vars)
    train_params = conf['training_parameters']
    decoupled_mode = train_params.get('decoupled_learning_rate', False)  # 默认为 False
    base_lr = train_params.get('learning_rate')
    enc_lr = train_params.get('encoder_learning_rate')
    if decoupled_mode:
        # decouple lr
        if base_lr is None:
            raise ValueError(
                "Config Error: 'decoupled_learning_rate' is True, but 'learning_rate' (decoder) is missing.")

        if enc_lr is None:
            raise ValueError("Config Error: 'decoupled_learning_rate' is True, but 'encoder_learning_rate' is missing!")

        print(f"[Config] strict mode: Decoupled LR Enabled. Encoder: {enc_lr}, Decoder: {base_lr}")
    else:
        if base_lr is not None and enc_lr is None:
            enc_lr = base_lr
        elif base_lr is None and enc_lr is not None:
            base_lr = enc_lr
        elif base_lr is None and enc_lr is None:
            raise ValueError("Error: No learning rate specified in config.")

        if base_lr != enc_lr:
            print(
                f"[Config] Warning: 'decoupled_learning_rate' is False/Missing, but separate LRs were provided. Forcing them to execute as separate LRs.")

    conf['training_parameters']['learning_rate'] = base_lr
    conf['training_parameters']['encoder_learning_rate'] = enc_lr

    if cl_vars['overfit'] or conf['run_type']['training_mode']=='overfit':
        conf['training_parameters']['validate_every'] = 100
        conf['training_parameters']['val_metric']['metric'] = 'train_loss'
        conf['training_parameters']['val_metric']['mode'] = 'min'
        conf['callbacks']['early_stopping']['monitor'] = 'train_loss'
        conf['callbacks']['early_stopping']['mode'] = 'min'
    if cl_vars['overfit'] or cl_vars['debug'] or conf['run_type']['training_mode']=='overfit' or conf['run_type']['training_mode']=='debug': 
        shuffle = False
    else:
        shuffle = True
    if conf['run_type']['pretrained_model'] is None:
        conf['run_name'] = cl_vars['run_name']
    else:
        conf['run_name'] = conf['run_type']['pretrained_model']
    dinov3_conf = conf.get('dinov3_integration', {'enabled': False})
    use_dinov3 = dinov3_conf.get('enabled', False)
    ft_conf = conf['training_parameters'].get('target_finetuning', {})
    ft_enabled = ft_conf.get('enabled', False)

    if use_dinov3:
        features_path = dinov3_conf.get('features_path')
        if not features_path:
            features_path = os.path.join(conf['paths']['output_dir'], 'dino_features')
            print(f"DINOv3 features_path not specified, defaulting to: {features_path}")
        conf['dinov3_integration']['features_path'] = features_path
        os.makedirs(features_path, exist_ok=True)
        dino_repo_path = conf['paths'].get('dinov3_repo_path')
        model_key = dinov3_conf.get('model_key')
        if not dino_repo_path or not model_key:
            raise ValueError("DINOv3 is enabled, but 'dinov3_repo_path' in paths or 'model_key' in dinov3_integration is missing in the config file.")
        
        model_config = conf.get('model_zoo', {})
        if not model_config:
            raise ValueError(f"Model key '{model_key}' not found in 'model_zoo' section of the config file. Aborting...")
        model_config_json = json.dumps(model_config)
        cache_dir = conf['paths'].get('torch_cache_dir')

        # kl_weight = ft_conf.get('kl_weight', 0.0) if ft_enabled else 0.0

        if not os.listdir(features_path):
            print("\n" + "="*80)
            print(f"DINOv3 features not found in {features_path}. Starting one-time preprocessing...")
            print("This may take a while, but only needs to be done once.")
            preprocess_script_path = os.path.join(os.path.dirname(__file__), 'preprocess_features.py')
            command = [
                sys.executable, 
                preprocess_script_path,
                '--csv_path', conf['paths']['training_csv'],
                '--output_dir', features_path,
                '--norm_file', conf['paths']['normalization_file'],
                '--dino_repo_path', dino_repo_path,
                '--model_config_json', model_config_json
            ]
            if cache_dir:
                command.extend(['--cache_dir', cache_dir])
            try:
                subprocess.run(command, check=True)
                print("Preprocessing finished successfully.")
            except subprocess.CalledProcessError as e:
                print(f"FATAL: Preprocessing script failed with exit code {e.returncode}. Aborting training.")
                sys.exit(1)
            except FileNotFoundError:
                print(f"FATAL: Preprocessing script not found at {preprocess_script_path}. Aborting.")
                sys.exit(1)
            print("="*80 + "\n")
        else:
            print(f"Found existing DINOv3 features in {features_path}. Skipping preprocessing.")
        #freeze other ddp processes here until dino features have been generated

        if ft_enabled:
            target_feat_path = ft_conf.get('target_dino_feature_path')
            if not target_feat_path:
                target_feat_path = os.path.join(conf['paths']['output_dir'], 'target_finetuning_dino_features')
                print(f"[Target Finetuning] feature path not specified. Defaulting to: {target_feat_path}")

            conf['training_parameters']['target_finetuning']['target_dino_feature_path'] = target_feat_path
            os.makedirs(target_feat_path, exist_ok=True)

            if not os.listdir(target_feat_path):
                print("\n" + "-" * 80)
                print(f"[Target Dataset] DINOv3 features not found in {target_feat_path}. Generating...")

                ft_csv_file = ft_conf.get('csv_file')
                if not ft_csv_file:
                    raise ValueError("Target finetuning enabled but 'csv_file' is missing!")

                ft_command = [
                    sys.executable,
                    preprocess_script_path,
                    '--csv_path', ft_csv_file,
                    '--output_dir', target_feat_path,
                    '--norm_file', conf['paths']['normalization_file'],
                    '--dino_repo_path', dino_repo_path,
                    '--model_config_json', model_config_json
                ]
                if cache_dir: ft_command.extend(['--cache_dir', cache_dir])

                try:
                    subprocess.run(ft_command, check=True)
                    print("Target dataset features generated.")
                except subprocess.CalledProcessError as e:
                    print(f"FATAL: Target Preprocessing failed. {e}")
                    sys.exit(1)
                print("-" * 80 + "\n")
            else:
                print(f"Found existing Target DINOv3 features in {target_feat_path}.")

    RUN_NAME = create_run_name(conf)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping_callback = EarlyStopping(patience=conf['callbacks']['early_stopping']['patience'], 
                                            monitor=conf['training_parameters']['val_metric']['metric'],
                                            mode=conf['training_parameters']['val_metric']['mode'])
    checkpoint_callback = ModelCheckpoint(monitor=conf['training_parameters']['val_metric']['metric'],
                                          mode=conf['training_parameters']['val_metric']['mode'],
                                          save_last=conf['callbacks']['model_checkpoint']['save_last'],
                                          save_weights_only=conf['callbacks']['model_checkpoint']['save_weights_only'],
                                          save_top_k=conf['callbacks']['model_checkpoint']['save_top_k'],
                                          filename=RUN_NAME+'-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}-{val_f1:.4f}')
    stats_saving_callback = StatsSavingCallback(save_every=conf['callbacks']['stats_saving']['save_every'], 
                                                save_hyperparameters=conf['callbacks']['stats_saving']['save_hyperparameters'])
    save_used_config_callback = ConfigSavingCallback(used_config=conf, config_filepath=cl_vars['config_file'])
    inspect_predictions = InspectPredictionsCallback(save_every=conf['training_parameters']['visualize_predictions_every'],
                                                     norm_file=conf['paths']['normalization_file'],
                                                     n_batches=conf['training_parameters']['visualize_n_batches'])
    callbacks = [checkpoint_callback, save_used_config_callback, lr_monitor, early_stopping_callback,
                 stats_saving_callback, inspect_predictions]
    # new development
    progress_bar_refresh_rate = conf['training_parameters']['progress_bar_refresh_rate']
    progress_bar_callback = TQDMProgressBar(refresh_rate=progress_bar_refresh_rate)
    callbacks.append(progress_bar_callback)

    if not ft_enabled and conf['training_parameters']['scheduler']['name'] in ['Exponential', 'OneCycleLR']:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    elif not ft_enabled and conf['training_parameters']['scheduler']['name'] in ['Plateau', 'CosineWarmRestarts','MultistepLR']:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if conf['training_parameters']['stochastic_weight_averaging']:
        swa_decoder_lr = base_lr * 0.1
        swa_encoder_lr = enc_lr * 0.1
        swa_lrs_list = [swa_decoder_lr, swa_encoder_lr]#these two should be exact match with encoder/decoder params
        callbacks.append(StochasticWeightAveraging(
            swa_epoch_start=0.8,
            annealing_strategy='cos',
            annealing_epochs=math.floor(0.2 * conf['training_parameters']['num_epochs']),
            swa_lrs=swa_lrs_list
        ))

    if conf['paths']['inference_dir'] is not None:
        predict_on_test = PredictOnTestSetCallback(save_every=conf['training_parameters']['predict_full_map_every'], 
                                                datapath=conf['paths']['inference_dir'], norm_file=conf['paths']['normalization_file'], 
                                                batch_size=conf['training_parameters']['batch_size'], 
                                                num_workers=conf['compute_parameters']['n_workers'], 
                                                tile_size=conf['training_parameters']['tile_size'])
        callbacks.append(predict_on_test)
    try:
        if conf['compute_parameters']['slurm_cluster']:
            callbacks.append(SLURMEnvironment(auto_requeue=True))
        print("Starting job on Slurm Cluster...")
    except Exception as Error:
        print('Running locally')

    finetuning_dataset_file = ft_conf.get('csv_file', None) if ft_enabled else None

    if ft_enabled:
        print(f"-> [Train] Fine-tuning ENABLED. Dataset: {finetuning_dataset_file}")
        # if kl_weight > 0:
        #     print(f"-> [Train] KL Divergence Regularization Enabled (Weight: {kl_weight})")
        conf['training_parameters']['scheduler']['name'] = None

    if conf['paths']['inference_dir'] is None:
        dblock = SatelliteDataModule(dataset_file=conf['paths']['training_csv'], 
                                     train_batch_size=conf['training_parameters']['batch_size'], 
                                     num_workers=conf['compute_parameters']['n_workers'],
                                     norm_file=conf['paths']['normalization_file'],
                                     shuffle=shuffle,
                                     tile_size=conf['training_parameters']['tile_size'],
                                     foldID=conf['run_type']['foldID'],
                                     image_type=conf['training_parameters']['image_type'],
                                     overfit_mode=conf['run_type']['training_mode']=='overfit',
                                     ssp = conf['run_type']['self_supervised_pretraining'],
                                     label_noise=conf['training_parameters']['label_noise'],
                                     in_ram_dataset=conf['training_parameters']['in_ram_dataset'],
                                     #regular domain adaptation
                                     adaption_task_dataset_file=conf['paths']['domain_adaptation_csv'],
                                     # dino
                                     use_dinov3_features=use_dinov3,
                                     dino_features_path=conf.get('dinov3_integration', {}).get('features_path'),
                                     dino_feature_dim=conf.get('model_zoo', {}).get(dinov3_conf.get('model_key'), {}).get('feature_dim', 1024),
                                     dino_patch_size=conf.get('model_zoo', {}).get(dinov3_conf.get('model_key'), {}).get('patch_size', 16),
                                     # one-shot finetune
                                     finetuning_dataset_file=finetuning_dataset_file,
                                     finetuning_config=ft_conf
                                     )
    else:
        dblock = SatelliteDataModule(dataset_file=conf['paths']['training_csv'], 
                                     test_datapath=conf['paths']['inference_dir'],
                                     train_batch_size=conf['training_parameters']['batch_size'], 
                                     norm_file=conf['paths']['normalization_file'],
                                     test_batch_size=conf['training_parameters']['batch_size'], 
                                     num_workers=conf['compute_parameters']['n_workers'],
                                     shuffle=shuffle,
                                     tile_size=conf['training_parameters']['tile_size'],
                                     foldID=conf['run_type']['foldID'],
                                     image_type=conf['training_parameters']['image_type'],
                                     overfit_mode=conf['run_type']['training_mode']=='overfit',
                                     ssp = conf['run_type']['self_supervised_pretraining'],
                                     label_noise=conf['training_parameters']['label_noise'],
                                     in_ram_dataset=conf['training_parameters']['in_ram_dataset'],
                                     # regular domain adaptation
                                     adaption_task_dataset_file=conf['paths']['domain_adaptation_csv'],
                                     #dino
                                     use_dinov3_features=use_dinov3,
                                     dino_features_path=conf.get('dinov3_integration', {}).get('features_path'),
                                     dino_feature_dim=conf.get('model_zoo', {}).get(dinov3_conf.get('model_key'), {}).get('feature_dim', 1024),
                                     dino_patch_size=conf.get('model_zoo', {}).get(dinov3_conf.get('model_key'), {}).get('patch_size', 16),
                                     #one-shot finetune
                                     finetuning_dataset_file=finetuning_dataset_file,
                                     finetuning_config=ft_conf
                                     )

    print("Tiles per epoch:", dblock.tiles_per_epoch, ", batch_size:", conf['training_parameters']['batch_size'],", Grad accumulation steps:", conf['training_parameters']['grad_acc_steps'])
    if not ft_enabled and conf['training_parameters']['scheduler']['name'] == 'OneCycleLR':
        conf['training_parameters']['scheduler']['params']['steps_per_epoch'] = math.ceil(dblock.tiles_per_epoch/(conf['training_parameters']['grad_acc_steps']*conf['training_parameters']['batch_size']))
        print("Steps per epoch:", conf['training_parameters']['scheduler']['params']['steps_per_epoch'])
    max_steps = 2*math.ceil(dblock.tiles_per_epoch/(conf['training_parameters']['grad_acc_steps']*conf['training_parameters']['batch_size']))*conf['training_parameters']['num_epochs']
    max_epochs = conf['training_parameters']['num_epochs']
    if conf['paths']['domain_adaptation_csv'] is not None:
        max_steps *= 2
    
    model_params = conf['model_parameters']
    model_name = conf['run_type']['pretrained_model']

    if use_dinov3:
        model_key = conf.get('dinov3_integration', {}).get('model_key')
        if model_key:
            model_params['model_config'] = conf.get('model_zoo', {}).get(model_key)
            model_params['repo_path'] = conf.get('paths', {}).get('dinov3_repo_path')
            model_params['cache_dir'] = conf.get('paths', {}).get('torch_cache_dir')

    model =  ModelTrainingWrapper(training_params=conf['training_parameters'], 
                                  model_params=model_params,
                                  from_pretrained=model_name,
                                  training_mode=conf['run_type']['training_mode'],
                                  domain_loss_scaling_factor=conf['training_parameters']['domain_loss_scaling_factor'],
                                  domain_classifier=conf['run_type']['domain_adaptation']
                                  # kl_scaling_factor=kl_weight
                                  )

    if conf['run_type']['from_checkpoint'] or cl_vars['checkpoint_file']:
        if conf['paths']['checkpoint_file'] is not None:
            ckpt_path = conf['paths']['checkpoint_file']
            if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
                print('-' * 100)
                print(f"Loading weights manually from: {ckpt_path}")
                print("NOTE: Optimizer state (LR) from checkpoint is IGNORED. Using config LR.")

                try:
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    print("Weights loaded successfully.")
                except Exception as e:
                    print(f"Error loading checkpoint weights: {e}")
                    sys.exit(1)

                print('-' * 100)
                resume_from_checkpoint = None
            else:
                print("No checkpoint found. Exiting...")
                sys.exit(1)
        else:
            print("Flag 'from_checkpoint' set to True but no checkpoint path provided. Exiting...")
            sys.exit(1)
    else:
        resume_from_checkpoint = None

    if cl_vars['overfit'] or conf['run_type']['training_mode']=='overfit':
        callbacks = [lr_monitor,]
        overfit_batches = True
        fast_dev_run = False
    else:
        callbacks = callbacks
        overfit_batches = False
        fast_dev_run = cl_vars['debug']
    ######### NEW DEVELOPMENT ############# 
    # YOU CAN LOAD THE MODEL WEIGHTS MANUALLY HERE AND IGNORE THE CHECKPOINT COMMAND LATER


    #######################################
    print("SAVING PATH", conf['paths']['output_dir']+'/experiments/'+conf['run_name'])
    print("Creating directory: ",conf['paths']['output_dir']+'/experiments/'+conf['run_name']+"/logs/")
    os.makedirs(conf['paths']['output_dir']+'/experiments/'+conf['run_name']+"/logs/", exist_ok=True)
    logger.addHandler(logging.FileHandler(conf['paths']['output_dir']+'/experiments/'+conf['run_name']+"/logs/"+"ExecutionLog{}.log".format(str(time.time()).split(',')[0])))
    
    # csv_logger = CSVLogger(
    #     save_dir=conf['paths']['output_dir'] + '/experiments/' + conf['run_name'],
    #     name='lightning_metrics_csv'
    # )
    # tb_logger = TensorBoardLogger(
    #     save_dir=conf['paths']['output_dir'] + '/experiments/' + conf['run_name'],
    #     name='lightning_metrics_tb'
    # )
    # tb_logger = TensorBoardLogger(save_dir=None,name=RUN_NAME, default_hp_metric=False, prefix="PerformanceLogs")

    if cl_vars['lr_find'] or conf['run_type']['training_mode']=='lr_find':
        lr_finder(model, dblock, conf, RUN_NAME, resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer = Trainer(
                        # progress_bar_refresh_rate=conf['training_parameters']['progress_bar_refresh_rate'], 
                        log_every_n_steps=conf['callbacks']['stats_saving']['save_every'],
                        max_steps = max_steps,
                        max_epochs = max_epochs,
                        default_root_dir=conf['paths']['output_dir']+'/experiments/'+conf['run_name'],
                        callbacks= callbacks, 
                        accumulate_grad_batches=conf['training_parameters']['grad_acc_steps'],
                        check_val_every_n_epoch=conf['training_parameters']['validate_every'],
                        resume_from_checkpoint=resume_from_checkpoint,
                        limit_train_batches=1 if overfit_batches else 1.0,
                        limit_val_batches=0 if overfit_batches else 1.0,
                        accelerator=conf['compute_parameters']['accelerator'],
                        gpus=conf['compute_parameters']['gpus'],
                        auto_select_gpus=conf['compute_parameters']['auto_select_gpus'],
                        num_nodes=conf['compute_parameters']['n_nodes'],
                        deterministic=conf['compute_parameters']['deterministic'],
                        fast_dev_run=fast_dev_run,
                        profiler="simple" if conf['run_type']['profile'] else None)
                        # logger=[csv_logger, tb_logger])
                        # max_epochs=conf['training_parameters']['num_epochs'], 
                        # logger=[tb_logger] )
                        # if accelerator == 'ddp': plugins=DDPPlugin(find_unused_parameters=False) 
                        # weights_save_path=conf['paths']['output_dir']
    # save upadated config (if some parameters have been over-writen by command line arguments)
    if not cl_vars['from_config']:
        os.makedirs(trainer.log_dir, exist_ok=True)
        yml = ruamel.yaml.YAML()
        yml.indent(mapping=4, sequence=4, offset=2)
        yml.preserve_quotes = True
        with open(trainer.log_dir+'/updated_config.yml', 'w') as fconf:
            yml.dump(conf, fconf)
    seed_everything(seed=conf['training_parameters']['random_seed'], workers=True)
    t1 = time.time()
    print( "-"*100,"\nSTARTING TRAINING...\n", "-"*100)
    trainer.fit(model, dblock)
    t2 = time.time()
    print(" FINISHED TRAINING\n","#"*80,"\n Total training time:", format((t2-t1)/60,'10.2f'),"m")
    ####### development
    if conf['paths']['inference_dir'] is not None:
        print( "-"*100,"\nSTARTING TESTING...\n", "-"*100)
        print(checkpoint_callback.best_model_path)
        Loader = TestingDataLoader( dataset_file=conf['paths']['inference_dir'], 
                                    batch_size=conf['training_parameters']['batch_size']//4, 
                                    norm_file=conf['paths']['normalization_file'], 
                                    tile_size=conf['training_parameters']['tile_size'], 
                                    split_tiles=False, TTA=False, 
                                    image_type=conf['training_parameters']['image_type'],
                                    use_only_test_tiles=False,
                                    num_workers=conf['compute_parameters']['n_workers'])
        testDataLoader = Loader.get_dataloader()
        results = trainer.test(dataloaders=testDataLoader, ckpt_path=trainer.checkpoint_callback.best_model_path)
        print("Evaluation results:")
        print(results)
        with open(trainer.log_dir+'end_of_training_evaluation.json', 'w') as jsonout:
            json.dump(results, jsonout, indent=4)
    else:
        results = {}
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    groupC = parser.add_argument_group("Configuration")
    groupC.add_argument('--from_config', action='store_true', help="Load all parameters from a config file.") 
    groupC.add_argument('-c', '--config_file', type=str, help="Config file to load parameters from. Without the \'from_config\' flag  parameters will be overwritten with defaults.", \
                        default=None) #'/home/minas/slumworld/slumworldML/src/base.yml')
    groupC.add_argument('--fold-id', type=int, help="If fold-id is -1 perform standard train-val-test training. Else load the fold-id fold from th k-fold cross validation dataset [default: -1]",
                        default=-1) 
    # groupC.add_argument('-n', '--normalization_file', type=str, help="File containing the image normalization stats (mean, std).", \
    #                     default='/global/scratch/users/mksifaki/data/tiled/MS2016-mul/info.yml') 
    groupC.add_argument('--logfile', type=str, help="File for logging program execution and warning messages.",
                        default='Execution_logs')
    groupC.add_argument('--run_name', type=str, required=False, help="Base filename to use for saving models (will add epoch number and best val_iou to it).",
                        default='unet')

    groupM = parser.add_argument_group("Vanila Unet Model Parameters")
    groupM.add_argument('--in_channels', type=int, required=False, default=None, help="Number of input channels, e.g. 3.") 
    groupM.add_argument('--out_channels', type=int, required=False, default=None, help="Number of output channels (classes per pixel), e.g. 2 for binary, 128 for signed distance.") 
    groupM.add_argument('--dropout', type=float, required=False, default=None, help="Dropout probability (if 0.0 no dropout is applied), e.g. 0.1.") 
    groupM.add_argument('--dropout2D', type=float, required=False, default=None, help="Dropout2D, i.e. probability of droping whole channels (if 0.0 no dropout2D is applied), e.g. 0.1.") 
    groupM.add_argument('--features', type=lambda s: [int(item) for item in s.split(',')], required=False, default=None, help="Comma separated list of the size of the encoder layers (decoder constructed symmetrically)"\
                        "of the UNET architecture, e.g. \'64,128,256,512\' -> 4 encoding layers with 64,128,256,512 feature maps respectivelly.") 
    
    groupT = parser.add_argument_group('Training Parameters')
    groupT.add_argument('--debug', action='store_true', help="Run only a single batch from each dataset (train, val, test) to verify operation.")
    groupT.add_argument('--pretrain', action='store_true', help="Turns on pretraining mode. Trains model in a self-supervised/ unsupervised fashion.")
    groupT.add_argument('--finetune', action='store_true', help="Turns on finetunning mode. Trains model with discriminative learning rates.")
    groupT.add_argument('--freeze', action='store_true', help="Freeze encoder and tune only decoder layers.")
    groupT.add_argument('--overfit', action='store_true', help="Overfit on a small portion of the training set (%%). Used for model testing. Monitors training loss.")
    groupT.add_argument('--lr_find', action='store_true', help="Runs the learning rate finder and saves an image of the results in the log_dir.")
    groupT.add_argument('--in_ram_dataset', action='store_true', help="Load all data in RAM (for faster loading - can lead to out of memory issues [default: false]).")
    groupT.add_argument('-p', '--pretrained_model', type=str, required=False, help="Name of the pre-trained (encoder) model to use, e.g. deeplabv3, unet_vgg11bn etc. If None (default) it constructs a standard unet). ",\
                        default=None)
    groupT.add_argument('--from_checkpoint', action='store_true', help="Continue training from checkpoint.")
    groupT.add_argument('--checkpoint_file', type=str, required=False, help="Full path to directory that contains the saved checkpoint.",\
                        default=None)
    groupT.add_argument('-d', '--training_csv', type=str, required=False, help="Full path to the directory that contains the training files. ",\
                        default=None)
    groupT.add_argument('-o', '--output_dir', type=str, required=False, help="Full path to directory to store all produced files (will be created if non-existent). ",\
                        default=None) #"/global/scratch/users/mksifaki/output")
    groupT.add_argument('-i', '--inference_dir', type=str, required=False, help="Full path to directory that contains the inference files (optional).",\
                        default=None)
    groupT.add_argument('--random_seed', type=int, required=False, help="Random seed to use for reproducibility, e.g. 1235", 
                        default=None)
    groupT.add_argument('--num_epochs', type=int, required=False, help="Number of epochs to train for, e.g. 1000.", 
                        default=None)
    groupT.add_argument('--learning_rate', type=float, required=False, help="Learning rate, e.g. 1e-3", 
                        default=None)
    groupT.add_argument('--batch_size', type=int, required=False, help="Batch size for the dataloader, e.g. 4. Actual batch_size will be the product of batch_size*num_accumulation_steps.", \
                        default=None)
    groupT.add_argument('--grad_acc_steps', type=int, required=False, help="Gradient accumulation steps, e.g. 5. Used to augment the batch_size in low memory settings. Enter 1 for no gradient accumulation", 
                        default=None)
    groupT.add_argument('--l2_reg', type=float, required=False, help="L2 regularization parameter, e.g. 1e-5.", 
                        default=None)
    groupT.add_argument('--criterion', type=str, required=False, help="Loss function to use, e.g. \'BinaryCrossEntropyWithLogits\'. See module help for options.", 
                        default=None)
    groupT.add_argument('--optimizer', type=str, required=False, help="Optimizer, e.g. \'Adam\'. See module help for options.", 
                        default=None)
    groupT.add_argument('--scheduler', type=str, required=False, help="Learning rate Scheduler, e.g. \'Plateau\'. See module help for options.", 
                        default=None)
    groupT.add_argument('--threshold', type=float, required=False, help="Probability threshold to use for selecting slums, typically 0.5.", 
                        default=None)

    groupP = parser.add_argument_group("Compute parameters")
    groupP.add_argument('--n_gpus', type=int, required=False, help="Number of gpus (per node) to train on, e.g. 2.",
                        default=None) 
    groupP.add_argument('--n_nodes', type=int, required=False, help="Number of nodes (each with n_gpus) for distributed training, e.g. 1.", 
                        default=None) 
    groupP.add_argument('--accelerator', type=str, required=False, help="One of dp (data parallel), ddp (distributed data parallel) acceleration for multi-gpu training or None.",
                        default=None) 

    args = parser.parse_args()

    model_trainer(vars(args))

    '''HOW TO RUN 
    >>> python3 train.py --config_file ./base.yaml --dropout_prob 0.5 
    >>> python3 train.py --config_file ./base.yaml --lr_find
    >>> python3 train.py --config_file ./base.yaml --overfit
    >>> python3 train.py --config_file ./base.yaml --debug
    >>> python3 train.py --config_file ./base.yaml --from_checkpoint # checkpoint path provided in yaml file
    HOW TO TUNE
    >>> python3 train.py --config_file ./base.yaml --parameter_to_tune1 value1 --parameter_to_tune2 value2 
    '''

    # TODO 
    # 1. add SWA
    # 2. add Augmentations
    # 3. add hierarchical parsing
    # 4. yml parsing + add normalization, + add transforms + add mixup
    # 5. print slum map every n epochs (full reconstruct)
    # 6. resume from checkpoint
    # 7. add n_gpus, n_cpus, accelerator, auto_select_gpus
    # CHECK: overfit, lr_find, scheduler 'interval':'step' parameter
