import sys
import os
import shutil
from pathlib import Path
import gc
import copy
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
import imageio
from PIL import Image
from skimage import io
import inspect
Image.MAX_IMAGE_PIXELS = None
import pdb


try:
        from slumworldML.src.model import MODELS_REGISTRY
        from slumworldML.src.utilities import *
        from slumworldML.src.SatelliteDataset import InferenceDataset
        from slumworldML.src.transforms_loader import create_transform, INFERENCE_TRANSFORMS_DICT
except Exception as Error:
    try:
        from src.model import MODELS_REGISTRY
        from src.utilities import *
        from src.SatelliteDataset import InferenceDataset
        from src.transforms_loader import create_transform, INFERENCE_TRANSFORMS_DICT
    except Exception as Error2:
        from model import MODELS_REGISTRY
        from utilities import *
        from SatelliteDataset import InferenceDataset
        from transforms_loader import create_transform, INFERENCE_TRANSFORMS_DICT
import pdb
class Predictor():
    """
    Usage:
        >>> import os
        >>> experiment = "/home/minas/slumworld/data/output/experiments/SatelliteUnet/lightning_logs/version_18/"
        >>> checkpoint = experiment + "checkpoints/BinaryCrossEntropyWithLogits-AdamW--L2_2e-05-Seed_1357-TStamp_1629139845-epoch=225-val_loss=0.0606-val_acc=0.9757-val_f1=0.7871808409690857.ckpt"
        >>> datapath = "/home/minas/slumworld/data/tiled/MD_MUL_75_Briana/"
        >>> norm_file = os.path.join(datapath, 'info.json')
        >>> predictor = Predictor()
        >>> predictor.load_model(model, model_path)
        >>> from SatelliteDataset import TestingDataLoader
        >>> Loader = TestingDataLoader(datapath, batch_size=25, norm_file=norm_file)
        >>> testDataLoader = Loader.get_dataloader()
        >>> predictor.load_data(testDataLoader)    # if you want to evaluate on a full dataset
        ,or, 
        >>> predictor.get_data_from_folder(datapath, batch_size=25, norm_file=norm_file) # for running inference directly on a folder 
                                                                                           of tiled images
        >>> results = predictor.evaluate()     # evaluate
        # visualize the predicted together with the input and the labels 
        >>> preds, labels, tilenames = predictor.predict_for_batches(number_of_batches=1, visualize=True)
        >>> pred = predictor.predict_on_images(img)        # predict on a single image
        >>> preds = predictor.predict_on_images(img_array) # predict on a batch of images
        # set a different threshold for classifying a slum (default = 0.5)
        predictor.threshold = 0.9
        >>> predictor.evaluate()
        # run inference on a full dataset (i.e. having no labels)
        # if no labels and/or tilenames are returned by the DataLoader then the corresponding returned lists will be empty
        >>> preds, labels, tile_names = predictor.predict_on_dataset()
    """
    def __init__(self, threshold=0.5):
        self.model_is_loaded = False
        self._threshold = threshold
        self.device = None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        """
        Modify the (softmax/sigmoid) threshold used to decide if a pixel is a slum [default:0.5]
        """
        self._threshold = threshold
    
    @staticmethod
    def get_freer_gpu():
        try:
            os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            return "cuda:" + str(np.argmax(memory_available))
        except Exception as Error:
            return "cuda:0"

    ## TODO: ADDED PARAMETER HAS CLASSIFIER HERE 
    def load_model(self, model, model_path, device="cpu", autoselect_gpu=True):
        if (device not in ['gpu','cpu']) and (not device.startswith('cuda')):
            print("Warning! Device should be gpu/cpu or cuda:{n}, provided strings is:", device)
            print("Switching to cpu...")
            device = 'cpu'
        if device=="cpu":
            self.device = device
        elif device=="gpu" or device.startswith('cuda'):
            if (self.device is None) and autoselect_gpu:
                self.device = self.get_freer_gpu()
            elif (self.device is None) and (not autoselect_gpu) and (not device.startswith('cuda:')):
                self.device = 'cuda:0'
            elif (self.device is None) and (not autoselect_gpu) and device.startswith('cuda:'):
                self.device = device
        elif device is None and not autoselect_gpu and (self.device is None):
            self.device = "cpu"
        print("Model loaded on divice:",self.device)
        self.model = model
        self.model_path = model_path
        self._load_model()

    def load_data(self, data_loader):
        self.data_loader = data_loader

    def _load_model(self):
        print("\nLoading trained model...")
        try:
            state_dict = torch.load(self.model_path, map_location=torch.device(self.device))
            # Handle checkpoints saved from PyTorch Lightning
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Create a new state_dict to filter out unnecessary keys from the wrapper
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    # Remove 'model.' prefix (e.g., 'model.encoder...' -> 'encoder...')
                    new_state_dict[k[len('model.'):]] = v
                elif not k.startswith('criterion.'):
                    # Keep weights that don't belong to the wrapper's criterion
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model_is_loaded = True
            self.model.eval()
            print("\nTrained model loaded on {}.".format(self.device))
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise e


    def _cleanup(self, verbose=False):
        try:
            if verbose: 
                print("Initially reserved memory:",torch.cuda.memory_reserved())
                print("Allocated memory:",torch.cuda.memory_allocated())
            self._model_to(torch.device('cpu'))
            del self.model
            gc.collect()
            self.model_is_loaded = False
            # torch.cuda.empty_cache()
            # self._dump_tensors()
            if verbose: 
                print("Reserved memory after cleanup:",torch.cuda.memory_reserved())
                print("Allocated memory:",torch.cuda.memory_allocated())
        except AttributeError:
            pass
    
    def _batch_predict(self, images, dino_features):
        # Check if the model is a DINOv3 variant by checking its forward signature
        model_forward_params = inspect.signature(self.model.forward).parameters

        if 'dino_features' in model_forward_params:
            # This is a dual-input model
            outputs = self.model(images, dino_features)
        else:
            # This is a single-input model, ignore dino_features
            outputs = self.model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        outputs_ = (torch.sigmoid(outputs) > self.threshold).long().detach().cpu().numpy()
        return outputs_

    def _batch_predict_proba(self, images, dino_features):
        """Returns raw sigmoid probabilities as float32 numpy array (no binarization)."""
        model_forward_params = inspect.signature(self.model.forward).parameters
        if 'dino_features' in model_forward_params:
            outputs = self.model(images, dino_features)
        else:
            outputs = self.model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return torch.sigmoid(outputs).detach().cpu().numpy().astype(np.float32)

    def predict_proba_for_batches(self):
        """Run full-dataset inference returning per-pixel sigmoid probabilities (not binarized).

        Returns:
            proba_list:  list of float32 np.arrays, one per batch, shape (B, 1, H, W)
            labels_list: list of int np.arrays, one per batch (empty list if dataset has no labels)
            tile_names:  list of name tuples, one per batch
        """
        if not self.model_is_loaded:
            self._load_model()
        proba_list, labels_list, tile_names = [], [], []
        with torch.no_grad():
            for batch_data in self.data_loader:
                labels = None
                if len(batch_data) == 4:
                    images, dino_features, labels, names = batch_data
                elif len(batch_data) == 3:
                    images, dino_features, names = batch_data
                else:
                    raise ValueError(
                        f"Unsupported batch format: expected 3 or 4 items, got {len(batch_data)}.")
                images_dev = images.to(self.device)
                dino_features_dev = dino_features.to(self.device)
                proba = self._batch_predict_proba(images_dev, dino_features_dev)
                proba_list.append(copy.deepcopy(proba))
                if labels is not None:
                    labels_list.append(copy.deepcopy(labels.numpy()))
                tile_names.append(copy.deepcopy(names))
        return proba_list, labels_list, tile_names

    @staticmethod
    def find_optimal_threshold(proba_list, labels_list, metric='f1',
                               threshold_range=(0.05, 0.95), threshold_step=0.05):
        """Search for the optimal binarization threshold on a labelled finetune dataset.

        Concatenates all pixels from all batches and evaluates every candidate threshold
        in one pass — this is fast because inference is only run once.

        Args:
            proba_list:      list of float32 np.arrays (B,1,H,W) from predict_proba_for_batches
            labels_list:     list of int np.arrays (B,*,H,W) ground-truth labels
            metric:          optimisation target: 'f1' or 'iou'
            threshold_range: (min, max) inclusive range of thresholds to search
            threshold_step:  step between candidate thresholds
        Returns:
            best_threshold:  float
            results_df:      pd.DataFrame with columns [threshold, f1, iou, precision, recall]
        """
        all_proba = np.concatenate([b.ravel() for b in proba_list])
        all_labels = np.concatenate([b.ravel() for b in labels_list]).astype(bool)

        thresholds = np.arange(
            threshold_range[0],
            threshold_range[1] + threshold_step * 0.5,
            threshold_step,
        )
        SMOOTH = 1e-6
        records = []
        for t in thresholds:
            pred = all_proba > t
            tp = np.logical_and(pred, all_labels).sum()
            fp = np.logical_and(pred, ~all_labels).sum()
            fn = np.logical_and(~pred, all_labels).sum()
            precision = tp / (tp + fp + SMOOTH)
            recall = tp / (tp + fn + SMOOTH)
            f1 = 2 * precision * recall / (precision + recall + SMOOTH)
            iou = tp / (tp + fp + fn + SMOOTH)
            records.append({
                'threshold': round(float(t), 4),
                'f1': float(f1),
                'iou': float(iou),
                'precision': float(precision),
                'recall': float(recall),
            })

        results_df = pd.DataFrame(records)
        score_col = 'f1' if metric == 'f1' else 'iou'
        best_threshold = float(results_df.loc[results_df[score_col].idxmax(), 'threshold'])
        return best_threshold, results_df

    def predict_for_batches(self, number_of_batches=-1, visualize=False, return_labels=True, return_names=True):
        if not self.model_is_loaded:
            self._load_model()

        if number_of_batches == -1:
            number_of_batches = len(self.data_loader)
        
        outputs_list, labels_list, tile_names = [], [], []
        data_iter = iter(self.data_loader)

        with torch.no_grad():
            for i in range(number_of_batches):
                # Unpack the items from the dataloader. The format depends on the dataset.
                batch_data = next(data_iter)
                
                # Default values
                images, dino_features, labels, names = None, None, None, None
                
                if len(batch_data) == 4: # e.g., ValidationDataset: (image, dino, label, name)
                    images, dino_features, labels, names = batch_data
                elif len(batch_data) == 3: # e.g., InferenceDataset: (image, dino, name)
                    images, dino_features, names = batch_data
                    return_labels = False # No labels available
                else:
                    raise ValueError(f"Unsupported batch format from DataLoader. Expected 3 or 4 items, got {len(batch_data)}.")

                # Move data to the correct device
                images_dev = images.to(self.device)
                dino_features_dev = dino_features.to(self.device)
                
                # Call the updated prediction method
                outputs = self._batch_predict(images_dev, dino_features_dev)
                
                outputs_list.append(copy.deepcopy(outputs))

                if return_labels and labels is not None:
                    labels_list.append(copy.deepcopy(labels.numpy()))

                if return_names and names is not None:
                    tile_names.append(copy.deepcopy(names))

                if visualize and labels is not None:
                    print(f"\nPrinting results for batch {i+1}:\n")
                    plot_images(images, labels, outputs)
        
        return outputs_list, labels_list, tile_names

    def predict_with_tta_loop(self, num_augs=5, USE_TTA=False, return_labels=True, return_names=True, release_memory=True):
        ''' Test Time Augmentation (TTA) loop method.
        Assumes that a dataloader with augmentations is loaded and calls predict for batches num_augs number of times, 
        collectes it's predictions and selects the majority prediction for each pixel.
        Parameter num_augs must be an ODD number (for a majority to always exist).
        ARGS:
            USE_TTA:        boolean, if set to False the method calls predict_for_batches
            num_aug:        int, the number of augmentation to produce (odd number) [default:5]
        Returns:
            ouptputs_list:  list[np.array], the predictions [num_batces x batch_size x 1 x width x height]
            and, depending on the kind of dataloader (validation, test, inference),
            labels_list:    list[np.array], the true labels (if existent) [num_batces x batch_size x 1 x width x height]
            tile_names:     list[str], the name of each tile [num_batces x batch_size x 1]
        NB: FOR THE METHOD TO WORK THE USER MUST PROVIDE A DATALOADER WITH TEST TIME AUGMENTATIONS
        '''
        # TODO verify that the dataloader has appropriate augmentations
        if USE_TTA:
            assert (num_augs%2 != 0) and (num_augs > 0), f"Error! Test time augmentation needs a odd number of augmentations in order to work. Supplied {num_augs}. Aborting ..."
            print("Warning! To use TTA (test time augmentations) it is assumed that you have supplied a dataloader with the transformations required.")
            print("This is the user's responsibility. No validation will take place.")
            for _ in range(num_augs):
                try:
                    outputs_list_i, labels_list, tile_names = self.predict_for_batches(number_of_batches=-1, 
                                                                                     visualize=False,
                                                                                     return_labels=return_labels,
                                                                                     return_names=return_names)
                    nargs = 3
                except Exception as Error:
                    try:
                        outputs_list_i, tile_names = self.predict_for_batches(number_of_batches=-1, 
                                                                            visualize=False,
                                                                            return_labels=return_labels,
                                                                            return_names=return_names)
                        nargs = 2
                    except Exception as Error:
                        outputs_list_i, = self.predict_for_batches(number_of_batches=-1, 
                                                                 visualize=False,
                                                                 return_labels=return_labels,
                                                                 return_names=return_names)
                        nargs = 1
                try:
                    outputs += np.vstack(outputs_list_i)
                except NameError:
                    outputs = np.vstack(outputs_list_i)
            # Get the most frequent prediction per pixel per tile
            outputs_list = [(outputs > 0.5 * num_augs).astype('uint8')]
            if release_memory:
                self._cleanup()
            if nargs == 3:
                labels_list = [np.vstack(labels_list)] 
                tile_names = [np.concatenate(tile_names)]
                return outputs_list, labels_list, tile_names
            elif nargs == 2:
                tile_names = [np.concatenate(tile_names)]
                return outputs_list, tile_names
            else:
                return outputs_list
        else:
            return self.predict_for_batches(number_of_batches=-1,visualize=False, 
                                            return_labels=True, return_names=True)


    def evaluate_model(self, save_tiles=False, output_path='.', predict_only=False, TTA=False, num_augs=5, dataset_csv=None):
        """
        Evaluates the trained model and plots the results.
        :args:
            output_path:        str, the folder where tiles were saved
            output_path:        str, the folder where tiles were saved
            save_tiles:         boolean, save prediction tiles to disk [default: False]
            output_path:        str, full pahth to the location that results should be saved [default:'.']
            predict_only:       boolean, if set to true the model will only perform predictions and save the results
                                (if save_tiles=True), else full evaluation will be run (required labels) [default:True]
            TTA:                boolean, Test Time Augmentation flag [default: false]
            num_augs:           int (odd), if TTA flag is set to True, this parameter will determine the number of 
                                augmentations to produce per tile
            dataset_csv:        str, the path to the dataset.csv file used for training/inference
                                (i.e. tiled_input, and tiled_labels folders)
        :return: 
            conf_matrix:            dict, confusion matrix (macro average) 
            metrics:                dict, metrics calculated (Accuracy, Precission, Recall, F1 Score, IOU)
            confusion_matrix_list:  list[dict], list of confusion matrices, per tile (if inspect_errros=True)
        """
        print("Predicting model outputs...")
        assert not self.model.training, "Error! Supplied model is in training mode. Evaluation will not be performed!"
        self.outputs_list, self.labels_list, self.tile_names = self.predict_with_tta_loop(num_augs=num_augs, USE_TTA=TTA)
        print("\nPredictions completed.")
        self._print_save_return_metrics(save_tiles=save_tiles, output_path=output_path, filename='evaluation.txt', 
                                        predict_only=predict_only, dataset_csv=dataset_csv)


    def _print_save_return_metrics(self, save_tiles, output_path, filename='evaluation.txt', predict_only=False, fraction=0.2,
                                   dataset_csv=None, create_new_dir=True, release_memory=True):
        '''Helper function for saving tiles, calculating performance metrics, error analysis and saving evaluatin results'''
        if create_new_dir:
            output_path = self._safely_create_dir(dirpath=output_path, overwrite=False)
        else:
            os.makedirs(output_path, exist_ok=True)

        if save_tiles:
            self.save_tiles(images=self.outputs_list, names=self.tile_names, output_path=output_path)
        if not predict_only:
            print("\nCalculating evaluation metrics:\n")
            confusion_matrices, micro_metrics_dict = compute_micro_metrics(true=self.labels_list, pred=self.outputs_list)
            conf_matrix, conf_matrix_printable, metrics, metrics_printable = compile_micro_metrics(confusion_matrices, print_result=True)
            if Path(output_path).is_dir():
                output_path = os.path.join(output_path, filename)
            with open(output_path, "w") as fout: 
                fout.write(conf_matrix_printable+'\n'+metrics_printable)
            try:
                if len(self.tile_names[-1]) != len(self.tile_names[-2]):        # last batch is smaller
                    diff = abs(len(self.tile_names[-2]) - len(self.tile_names[-1]))
                    last_batch = [t for t in self.tile_names[-1]]
                    last_batch.extend(['zero_padded']*diff)                     # pad to full length
                    self.tile_names[-1] = tuple(last_batch)
            except IndexError:          # there is only one batch, do nothing
                pass
            micro_metrics_dict['names'] = np.array(self.tile_names).ravel()
            metrics_df = pd.DataFrame.from_dict(micro_metrics_dict)
            metrics_df.sort_values(inplace=True, by=['f_one'], ascending=True)
            metrics_df['CDF_percentage'] = [float(f)/len(metrics_df) for f in range(1, len(metrics_df)+1)]
            if not Path(output_path).is_dir():
                output_path = Path(output_path).parent
            metrics_df.to_csv(os.path.join(output_path,'metrics_distribution.csv'))
            if dataset_csv is not None:
                try:
                    os.path.exists(dataset_csv)
                    save_tile_overlays(image_name_list=metrics_df.names[:int(fraction*len(metrics_df))].tolist(), 
                                    pred_path= output_path,
                                    dataset_csv= dataset_csv, 
                                    output_path=output_path)
                except Exception as Error:
                    print("Error! Could not find dataset_csv file. Filename given is:", dataset_csv)
            else:
                print("For error inspection the user needs to provide the full path to the dataset.csv file used for training.")
            if release_memory:
                self._cleanup()
            return conf_matrix, metrics
        else:
            self._cleanup()
        if save_tiles:
            return output_path

    def run_inference(self, save_tiles=True, output_path='.', TTA=False, num_augs=5):
        """
        Run inference using the trained model and (optionally) save the results.
        :args:
            save_tiles:         boolean, save prediction tiles to disk [default: False]
            output_path:        str, full pahth to the location that results should be saved [default:'.']
            TTA:                boolean, Test Time Augmentation flag [default: false]
            num_augs:           int (odd), if TTA flag is set to True, this parameter will determine the number of 
                                augmentations to produce per tile
        :return: 
            output_path:        str, the folder where tiles were saved
        """
        print("Predicting model outputs...")
        assert not self.model.training, "Error! Supplied model is in training mode. Evaluation will not be performed!"
        self.outputs_list, self.tile_names = self.predict_with_tta_loop(num_augs=num_augs, USE_TTA=TTA)
        print("\nPredictions completed.")
        if save_tiles:
            output_path = self._safely_create_dir(dirpath=output_path, overwrite=False)
            self.save_tiles(images=self.outputs_list, names=self.tile_names, output_path=output_path)
        return output_path

    ## TO TEST:
    def evaluate_model_ensemble(self, models_list=None, model_checkpoints=None, strategy='majority',
                                save_tiles=False, output_path='.', predict_only=False, device='cpu',
                                filename='ensemble_evaluation.txt', TTA=False, num_augs=5, dataset_csv=None, 
                                release_memory=True,dinov3_config=None):
        """
        Evaluates an ensemble of trained models and plots the results.
        :args:
            models_list:        list[str], list of model types
            model_checkpoints:  list[str], list of checkpoints of models (one for each of the model_types)
            strategy:           str, strategy for calculating outputs of the ensemble [default: 'majority']
            save_tiles:         boolean, save prediction tiles to disk [default: False]
            output_path:        str, full path to the location that results should be saved [default:'.']
            filename:           str, filename for the evalatuation results text file [default: 'evaluation.txt']
            predict_only:       boolean, if set to true the model will only perform predictions and save the results
                                (if save_tiles=True), else full evaluation will be run (required labels) [default:True]
            device:             str, one of 'cpu', 'gpu' the device to use for calculations [default: 'cpu']
            TTA:                boolean, Test Time Augmentation flag [default: false]
            num_augs:           int (odd), if TTA flag is set to True, this parameter will determine the number of 
                                augmentations to produce per tile
            inspect_errors:     bool, if set to True the function will return a list of confusion maps (one per each tile)
                                for further processing and error analysis
        :return: 
            conf_matrix:            dict, confusion matrix (macro average) 
            metrics:                dict, metrics calculated (Accuracy, Precission, Recall, F1 Score, IOU)
            confusion_matrix_list:  list[dict], list of confusion matrices, per tile (if inspect_errros=True)
        """
        print("Predicting model outputs...")
        all_outputs = []
        for i, (model_name, checkpoint_path) in enumerate(zip(models_list, model_checkpoints)):
            print(f"\nPredicting for model {i+1}, type: {model_name}")
            assert model_name in MODELS_REGISTRY.keys(), f"Model type not in available model types: {MODELS_REGISTRY.keys()}"
            
            model_class = MODELS_REGISTRY[model_name]
            # Assume models for ensembling don't need DINOv3 for now, or this needs to be passed from config
            # This is a safe default for non-DINOv3 models in your registry
            if 'dinov3' in model_name.lower() and dinov3_config:
                print(f"Instantiating DINOv3 model '{model_name}' with custom config.")
                model_instance = model_class(
                    in_channels=3, 
                    out_channels=1, 
                    domain_classifier=False,
                    dino_feature_dim=dinov3_config.get('feature_dim', 1024)
                )
            else:
                print(f"Instantiating standard model '{model_name}'.")
                model_instance = model_class(in_channels=3, out_channels=1, domain_classifier=False)
            
            self.load_model(model_instance, checkpoint_path, device=device)
            
            # The predict_with_tta_loop call now correctly returns 3 values
            outputs_list_i, labels_list_i, tile_names_i = self.predict_with_tta_loop(
                num_augs=num_augs, USE_TTA=TTA, return_labels=(not predict_only), return_names=True
            )
            
            all_outputs.append(np.vstack(outputs_list_i))
            if not predict_only:
                self.labels_list = labels_list_i # All label/name lists should be the same
            self.tile_names = tile_names_i
            
            if release_memory:
                self._cleanup()

        # Ensemble logic (majority vote)
        ensembled_outputs = (np.sum(all_outputs, axis=0) > len(models_list) / 2).astype(np.uint8)
        self.outputs_list = [ensembled_outputs]
        
        if not predict_only:
            self.labels_list = [np.vstack(self.labels_list)]
        self.tile_names = [np.concatenate(self.tile_names)]
        
        print("\nEnsemble predictions completed.")
        return self._print_save_return_metrics(save_tiles=save_tiles, output_path=output_path, filename=filename, 
                                               predict_only=predict_only, dataset_csv=dataset_csv)

    def run_inference_with_ensemble(self, models_list=None, model_checkpoints=None, strategy='majority',
                                    save_tiles=False, output_path='.', device='cpu', TTA=False, num_augs=5,
                                    release_memory=True):
        
        """
        Run inference using an ensemble of trained model and (optionally) save the results.

        :args:
            models_list:        list[str], list of model types
            model_checkpoints:  list[str], list of checkpoints of models (one for each of the model_types)
            strategy:           str, strategy for calculating outputs of the ensemble [default: 'majority']
            save_tiles:         boolean, save prediction tiles to disk [default: False]
            output_path:        str, full path to the location that results should be saved [default:'.']
            device:             str, one of 'cpu', 'gpu' the device to use for calculations [default: 'cpu']
            TTA:                boolean, Test Time Augmentation flag [default: false]
            num_augs:           int (odd), if TTA flag is set to True, this parameter will determine the number of 
                                augmentations to produce per tile
        :return: 
            output_path:        str, the folder where tiles were saved (if save_tiles=True)
        """
        print("Predicting model outputs...")
        i = 0
        for model_i, checkpoint_i in zip(models_list, model_checkpoints):
            print(f"Predicting for model {i+1}, type: {model_i}")
            assert model_i in MODELS_REGISTRY.keys(), f"Model type not in available model types: {MODELS_REGISTRY.keys()}"
            self.load_model(MODELS_REGISTRY[model_i](in_channels=3,
                                                     out_channels=1,
                                                     domain_classifier=False
                                                     ), checkpoint_i, device=device) 
            outputs_list_i, self.tile_names = self.predict_with_tta_loop(num_augs=num_augs, USE_TTA=TTA)
            try:
                outputs += np.vstack(outputs_list_i)
            except NameError:
                outputs = np.vstack(outputs_list_i)
            i += 1            
            if release_memory:
                print("Releasing memory....")
                self._cleanup()
        self.tile_names = [np.concatenate(self.tile_names)]
        if strategy == 'majority':
            # Get the most frequent prediction per pixel per tile
            self.outputs_list = [(outputs > 0.5 * i).astype('uint8')]
        print("\nPredictions completed.")
        output_path = self._safely_create_dir(dirpath=output_path, overwrite=False)
        if save_tiles:
            self.save_tiles(images=self.outputs_list, names=self.tile_names, output_path=output_path)
        return output_path


    def evaluate_from_full_maps(self, input_slum_map, predicted_slum_map, output_path=None, 
                                results_filename='evaluation_results_overlap.txt'):
        '''Calculate the evaluation metrics from full maps rather than tiles
        N.B.: Evaluating on a full map will produce DIFFERENT RESULTS than when evaluating
              on individual tiles since the reconstruction of the map involves removing the padding
              and hence the number of pixels in the reconstructed map is smaller than the sum of the number
              of pixesls of the tiles.
              
        :args:
            input_slum_map:         str, full path to the location of the true slum map
            predicted_slum_map:     str, full path to the location of full predicted sum map
            output_path:            str/None, if a path to a directory is provided then the results will be saved in file 'evaluation.txt'
                                    else they will only be returned by the function
        :return: 
            conf_matrix:            dict, confusion matrix (macro average) 
            metrics:                dict, metrics calculated (Accuracy, Precission, Recall, F1 Score, IOU)
            None:                   this argument is supplied for compatibility with other versions of the evaluation code
        :usage:
            >>> labels_img = "/home/minas/slumworld/data/raw/MD_MUL_97_2016_Brenda/inputs/input_y.png"
            >>> reconstructed_predictions_img = "/home/minas/slumworld/data/output/experiments/unet_vgg11_bn/lightning_logs/version_9/reconstructedNewMD.png"
            >>> predictor = Predictor()
            >>> predictor.evaluate_full_maps(labels_img, reconstructed_predictions_img)
        '''
        true_map = self.load_and_fix_slum_map(input_slum_map)
        predicted_map = self.load_and_fix_slum_map(predicted_slum_map)
        self.outputs_list = [predicted_map]
        self.labels_list = [true_map]
        self._print_save_return_metrics(save_tiles=False, output_path=output_path, filename=results_filename, 
                                        predict_only=False, create_new_dir=False)
   

    @staticmethod
    def _safely_create_dir(dirpath, overwrite=False):
        '''Before creating a directory check if it exists and based on the flag overwrite either delete
           and create it again or create a new directory with the same name with a '.1'/'.2'/... appended to its end.'''
        if not isinstance(dirpath, str):
            dirpath = str(dirpath)
        def _increase_dir_number(dirpath):
            while os.path.exists(dirpath):
                try:
                    n = int(dirpath.rsplit('.')[-1])
                    dirpath = '.'.join(dirpath.rsplit('.')[:-1]) + '.' + str(n+1)
                except ValueError:
                    dirpath += '.1'
            return dirpath
        if os.path.exists(dirpath):
            if overwrite:
                shutil.rmtree(dirpath)
            else:
                dirpath = _increase_dir_number(dirpath)
        try:
            os.makedirs(dirpath)                
        except FileExistsError:
            dirpath = _increase_dir_number(dirpath)
            os.makedirs(dirpath)  
        return dirpath

    @staticmethod
    def load_and_fix_slum_map(map_file):
        map = io.imread(map_file)
        if map.shape[-1] == 3:
            # 3d image
            map = map[:,:,0]
        elif map.shape[-1] == 1:
            # 3d image
            map = np.squeeze(map, axis=-1)
        if np.max(map) == 255:      # reconstructed slum map for visualization => binarize
            if len(np.unique(map)) <= 2:
                map = np.true_divide(map, 255).astype('uint8')
            else:
                print(f"Error! Predicted slum map file {map_file} can only be binary or have values of 0 and 255.")
                print("Aborting operation...")
                sys.exit(1)
        if np.max(map) == 127:     # signed distance map => binarize
            map[map<64] = 0
            map[map>=64] = 1
        return map.astype('uint8')

    #######################################################################
    # TODO

    def process_overlaped_predictions(self, overlaped_predictions):
        '''Process predictions from overlap tiles to extract (and optionally save) the central part AND label.'''
        pass
    #######################################################################


    def predict_on_images(self, images):
        # TODO 
        # ADD NORMALIZE
        if isinstance(images, list):
            images = np.stack(images, axis=0)

        if not (torch.is_tensor(images)): 
            images = torch.from_numpy(np.array(images)).float()

        if images.dim() <= 3:
            images = torch.unsqueeze(images, dim=0)

        with torch.no_grad():
            outputs = self._batch_predict(images)
            
        return outputs

    def predict_on_dataset(self):
        return self.predict_for_batches(number_of_batches=-1, visualize=False, return_labels=False, return_names=True)

    def save_tiles(self, images, names, output_path='.', format='png'):
        """
        Given a list of images, a list of appropriate names and a path to a directory to save to, this will save the images to disc.
        If the path directory does not exist, it will be created.
        """
        assert len(images) == len(names) > 0, print("The number of images must be equal to the number of filenames and must be larger than 0")
        output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(exist_ok=True, parents=True)
        if isinstance(images, np.ndarray):
            img_iterator = [images[i] for i in range(len(images))]
        elif isinstance(images, list):
            img_iterator = images
        else:
            print("Error! imgs must either be a list of numpy arrays or a 3/4 dimensional numpy array (dimension 0 being the batch)!")
            return 
        try:
            for img_batch_i, name_batch_i in zip(img_iterator, names):
                self.save_batch_images(img_batch_i, name_batch_i, base_path=output_path)
            print("Operation completed succesfully. Predicted tiles saved to folder:", str(output_path))
        except Exception as Error:
            print("Something went wrong.\n", Error)
        
    def save_batch_images(self, imgs_iterator, names_iterator, base_path=None, format:str='png'):
        for img_i, name_i in zip(imgs_iterator, names_iterator):
            im_i = self.imgmat2img(img_i)
            imageio.imwrite(uri=os.path.join(base_path, name_i), 
                            im=im_i, format=format)

    def imgmat2img(self, imgmat, IMGTYPE='uint8'):
        '''Helper function that makes sure we can save images of different channel depths.
        It will convert single channel images to 3-channel grayscales 
        and will keep the first 3 channels of 4 channel ones. 
        Will convert float values to 8bit ints.'''
        assert isinstance(imgmat, np.ndarray), "Image matrix not a numpy array"
        img_in = copy.deepcopy(imgmat)
        # img_in *= 255
        if len(img_in.shape) > 3:
            return img_in[:,:,:3].astype(IMGTYPE)
        if len(img_in.shape) < 3:
            img_in = np.atleast_3d(img_in)
        if img_in.shape[0] == 1:
            img_in = np.transpose(img_in,axes=(1,2,0))
        if img_in.shape[-1] == 1:
            return np.repeat(img_in, repeats=3, axis=2).astype(IMGTYPE)
        else:
            return img_in.astype(IMGTYPE)

    def get_data_from_folder(self, datafolder, normalization_file=None, transforms='inference_mul', tile_size=512, batch_size=25, num_workers=2):
        '''Create a data loader by fetching data from a folder (i.e. for pure inference mode on data
        from other sources, i.e. when no dataset.csv is available)
        :args:
            datafolder:         str, location of the tiles
            normalization_file: str, full path to a json file holding the normalization stats, i.e.:
                                {'mean':[x,y,z], 'std':[p,q,r]}
            transforms:         str, what type of transforms to apply, options are[ 'inference_mul', 'inference_pan', tta_mul','tta_pan']
                                [defaul:'inference_mul']
            tile_size:          int, tile_size [default: 512]
            batch_size:         int boolean, [default: 25]
            num_workers:        int, nummber of workers for loading data in parallel [default: 2]
        :populates:             the self.data_loader property
        '''
        if normalization_file is not None:
            try:
                with open(normalization_file, 'r') as json_file:
                    data = json.load(json_file)
                mean = data["mean"]
                std = data["std"]
            except Exception as Error:
                print("Error! Could not load normalization file. Aborting...", Error)
                sys.exit(1)
        else:
            print("Normalization File not Provided. Predictions will not be performed. Exiting...")
            sys.exit(1)
        infer_transforms = create_transform(INFERENCE_TRANSFORMS_DICT[transforms], mean=mean, std=std, input_size=tile_size)
        dataset = InferenceDataset(datafolder, tile_size=tile_size, transform=infer_transforms)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)

    def check_dataloader(self):
        print("Checking dataloader...")
        number_of_batches = len(self.data_loader)
        print("Number of batches:", number_of_batches)
        data_iter = iter(self.data_loader)
        tile_names = []
        with torch.no_grad():
            for i in range(number_of_batches):
                dat = next(data_iter)
                if len(dat) > 1:
                    names = dat[-1]
                else:
                    names = dat[0]
                tile_names.extend(names)
        print("Number of tiles:", len(list(set(tile_names))))

    @staticmethod
    def _pretty_size(size):
        """Pretty prints a torch.Size object"""
        assert(isinstance(size, torch.Size))
        return " × ".join(map(str, size))

    
    def _dump_tensors(self, gpu_only=True):
        """Prints a list of the Tensors being tracked by the garbage collector."""
        import gc
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if not gpu_only or obj.is_cuda:
                        print("%s:%s%s %s" % (type(obj).__name__, 
                                            " GPU" if obj.is_cuda else "",
                                            " pinned" if obj.is_pinned else "",
                                            self._pretty_size(obj.size())))
                        total_size += obj.numel()
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                    type(obj.data).__name__, 
                                                    " GPU" if obj.is_cuda else "",
                                                    " pinned" if obj.data.is_pinned else "",
                                                    " grad" if obj.requires_grad else "", 
                                                    " volatile" if obj.volatile else "",
                                                    self._pretty_size(obj.data.size())))
                        total_size += obj.data.numel()
            except Exception as e:
                pass        
        print("Total size:", total_size)

    def _model_to(self, device):
        for param in self.model.parameters():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
