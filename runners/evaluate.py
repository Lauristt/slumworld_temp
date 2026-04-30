''' Script for evaluating a trained model.
The script loads a trained model, evaluates it on a given dataset and saves the results.
Optionally it can also reconstruct the original map (if run on all image tiles).
It reads all required parameteres (experiment_path , checkpoint_path, data_path, 
norm_file_path, reconstruct_map, output_dir, model type ) from a configuration.yml file.
The only argument that needs to be passed to the script is the location of the configuration file.
ARGS:
    -c [--config_file]:             str, full path to the configuration file [default: \'evaluate.yml\']
USAGE:
    >>> python3 evaluate.py -c /path/to/evaluate.yml
'''
import sys
import os
import shutil
import warnings
import pdb
import datetime
import subprocess
from torchvision.transforms.functional import convert_image_dtype
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
import pandas as pd
from skimage import io
try:
    from slumworldML.src.model import MODELS_REGISTRY
    from slumworldML.src.predictor import Predictor
    from slumworldML.src.SatelliteDataset import TestingDataLoader
    from slumworldML.src.base_tiler import ImageTiler
    from slumworldML.src.overlap_tiler import OverlapTiler
    from slumworldML.src.inspector import overlay, generate_shapefiles
except Exception as Error:
    try:
        from src.model import MODELS_REGISTRY
        from src.predictor import Predictor
        from src.SatelliteDataset import TestingDataLoader
        from src.base_tiler import ImageTiler
        from src.overlap_tiler import OverlapTiler
        from src.inspector import overlay, generate_shapefiles
    except Exception as Error2:
        try:
            from .src.model import MODELS_REGISTRY
            from .src.predictor import Predictor
            from .src.SatelliteDataset import TestingDataLoader
            from .src.base_tiler import ImageTiler
            from .src.overlap_tiler import OverlapTiler
            from .src.inspector import overlay, generate_shapefiles
        except Exception as Error3:
            from ..src.model import MODELS_REGISTRY
            from ..src.predictor import Predictor
            from ..src.SatelliteDataset import TestingDataLoader
            from ..src.base_tiler import ImageTiler
            from ..src.overlap_tiler import OverlapTiler
            from ..src.inspector import overlay, generate_shapefiles


def generate_random_string_from_time():
    current_time = datetime.datetime.now()
    random_string = current_time.strftime("%Y%m%d%H%M%S%f")
    return random_string

def main(config, config_file):

    experiment_paths = [Path(p) for p in config['experiment_path']]
    checkpoint_paths = config['checkpoint_path']
    if config['ensemble_predictions']:
        checkpoints = [p/c for p,c in zip(experiment_paths, checkpoint_paths)]
    else:
        checkpoint = experiment_paths[0]/checkpoint_paths[0]
    if config['output_dir'] is not None:
        output_dir = Path(config['output_dir'])
    else:
        output_dir = experiment_paths[0]/'evaluation'
    dinov3_conf = config.get('dinov3_integration', {})

    # Enable DINOv3 features ONLY if the config enables it AND at least one model in the list is a DINOv3 model.
    is_dinov3_run = dinov3_conf.get('enabled', False) and any('dinov3' in m.lower() for m in config['model_type'])
    if is_dinov3_run:
        features_path = dinov3_conf.get('features_path')
        if not features_path:
            raise ValueError("DINOv3 is enabled, but 'features_path' is not specified in the dinov3_integration config.")

        os.makedirs(features_path, exist_ok=True)
        if not os.listdir(features_path):
            print("\n" + "="*80)
            print(f"DINOv3 features for evaluation not found in {features_path}. Starting one-time preprocessing...")
            dino_repo_path = dinov3_conf.get('dinov3_repo_path')
            model_key = dinov3_conf.get('model_key')
            if not dino_repo_path or not model_key:
                raise ValueError("DINOv3 is enabled, but 'dinov3_repo_path' in paths or 'model_key' in dinov3_integration is missing.")

            model_config = config.get('model_zoo', {}).get(model_key)
            if not model_config:
                raise ValueError(f"Model key '{model_key}' not found in 'model_zoo'.")

            model_config_json = json.dumps(model_config)
            cache_dir = config.get('paths', {}).get('torch_cache_dir')

            script_dir = os.path.dirname(os.path.abspath(__file__))
            preprocess_script_path = os.path.join(script_dir, 'preprocess_features.py') 

            command = [
                sys.executable,
                preprocess_script_path,
                '--csv_path', str(config['dataset_path']),
                '--output_dir', features_path,
                '--norm_file', str(config['normalization_file_path']),
                '--dino_repo_path', dino_repo_path,
                '--model_config_json', model_config_json
            ]
            if cache_dir:
                command.extend(['--cache_dir', cache_dir])

            try:
                subprocess.run(command, check=True)
                print("Preprocessing for evaluation finished successfully.")
            except subprocess.CalledProcessError as e:
                print(f"FATAL: Preprocessing script failed with exit code {e.returncode}. Aborting evaluation...")
                sys.exit(1)
            except FileNotFoundError:
                print(f"FATAL: Preprocessing script not found at {preprocess_script_path}. Aborting...")
                sys.exit(1)
            print("="*80 + "\n")
        else:
            print(f"Found existing DINOv3 features for evaluation in {features_path}. Skipping preprocessing.")
    dataset_path = Path(config['dataset_path'])
    norm_file = Path(config['normalization_file_path'])
    save_tiles = config['save_tiles']

    if config['image_type'] not in ["mul", "pan", "pans"]:
        print("Image type not understood. If not using one of mul/pan then define the number of input channels in the code.")
        print("Exiting...")
        sys.exit(1) 

    if config['ensemble_predictions']:
        models_list = config['model_type']
        if len(models_list)%2 == 0:
            
            print("Error! Selected model ensembling while providing an even number of models.")
            print("Ensembling is based on majority voting and hence requires an odd number of models.")
            print("Exiting...")
            sys.exit(2)
    else:
        model_i = config['model_type'][0]
        assert model_i in MODELS_REGISTRY.keys(), f"Model type not in available model types: {MODELS_REGISTRY.keys()}"
        model = MODELS_REGISTRY[model_i](in_channels=3, out_channels=1, domain_classifier=False)
    DEVICE = config['device']   

    output_path_ = Path(os.path.join(os.path.dirname(os.path.abspath(output_dir)), generate_random_string_from_time()))

    predictor = Predictor()
    predictor.threshold = config['threshold']
    
    if not config['overlapping_tiles']:
        df = pd.read_csv(str(dataset_path.absolute()))
        tile_size = io.imread(df.iloc[0,0]).shape[1]

        if not config['use_masked']:
            tmp_masked_folder = str((output_dir/'tmp_masked_folder').absolute())
            tmp_dataset_csv = os.path.join(tmp_masked_folder, os.path.split(str(dataset_path.absolute()))[-1])
            os.makedirs(tmp_masked_folder, exist_ok=True)
            df = df.drop(df[df['dataset_part']=='Mask'].index)
            df.to_csv(tmp_dataset_csv)
            dataset_path = Path(tmp_dataset_csv)
        Loader = TestingDataLoader(
            str(dataset_path.absolute()),
            batch_size=config['batch_size'],
            norm_file=str(norm_file.absolute()),
            tile_size=tile_size,
            split_tiles=config['split_tiles'],
            TTA=config['tta'],
            image_type=config['image_type'],
            use_only_test_tiles=config['use_only_test_tiles'],
            exclude_test_tiles=config.get('exclude_test_tiles', False),
            use_dinov3_features=is_dinov3_run,
            dino_features_path=dinov3_conf.get('features_path'),
            dino_feature_dim=dinov3_conf.get('feature_dim', 1024),
            dino_patch_size=dinov3_conf.get('patch_size', 16))

        testDataLoader = Loader.get_dataloader()

        predictor.load_data(testDataLoader)

        # ── Threshold fine-tuning (runs before evaluation for both ensemble and single-model) ──
        ft_conf = config.get('threshold_finetune', {})
        if ft_conf.get('enabled', False):
            ft_dataset_csv = ft_conf.get('dataset_csv')
            if not ft_dataset_csv or not Path(ft_dataset_csv).exists():
                print(f"Warning: threshold_finetune.dataset_csv not found ({ft_dataset_csv}). Skipping threshold fine-tuning.")
            else:
                print("\n" + "="*60)
                ft_df = pd.read_csv(str(Path(ft_dataset_csv).absolute()))
                ft_tile_size = io.imread(ft_df.iloc[0, 0]).shape[1]
                ft_loader = TestingDataLoader(
                    str(Path(ft_dataset_csv).absolute()),
                    batch_size=config['batch_size'],
                    norm_file=str(norm_file.absolute()),
                    tile_size=ft_tile_size,
                    split_tiles=config['split_tiles'],
                    TTA=False,
                    image_type=config['image_type'],
                    use_only_test_tiles=ft_conf.get('use_only_finetune_tiles', False),
                    use_dinov3_features=is_dinov3_run,
                    dino_features_path=dinov3_conf.get('features_path'),
                    dino_feature_dim=dinov3_conf.get('feature_dim', 1024),
                    dino_patch_size=dinov3_conf.get('patch_size', 16),
                )
                ft_dataloader = ft_loader.get_dataloader()

                if config['ensemble_predictions']:
                    # Collect per-model probabilities and average (soft voting)
                    print("Running threshold fine-tuning on finetune dataset (ensemble: averaging probabilities across models)...")
                    ft_proba_sum = None
                    ft_labels_list = None
                    for model_i_name, checkpoint_i in zip(models_list, checkpoints):
                        model_ft = MODELS_REGISTRY[model_i_name](in_channels=3, out_channels=1, domain_classifier=False)
                        predictor.load_model(model_ft, str(checkpoint_i.absolute()),
                                             device=DEVICE, autoselect_gpu=config['autoselect_gpu'])
                        predictor.load_data(ft_dataloader)
                        ft_proba_list_i, ft_labels_list_i, _ = predictor.predict_proba_for_batches()
                        proba_arr = np.vstack(ft_proba_list_i)
                        ft_proba_sum = proba_arr if ft_proba_sum is None else ft_proba_sum + proba_arr
                        if ft_labels_list is None and ft_labels_list_i:
                            ft_labels_list = ft_labels_list_i
                        predictor._cleanup()
                    ft_proba_list = [ft_proba_sum / len(models_list)]
                else:
                    # Single model: load once, collect probabilities
                    print("Running threshold fine-tuning on finetune dataset...")
                    predictor.load_model(model, str(checkpoint.absolute()),
                                         device=DEVICE, autoselect_gpu=config['autoselect_gpu'])
                    predictor.load_data(ft_dataloader)
                    ft_proba_list, ft_labels_list, _ = predictor.predict_proba_for_batches()

                if not ft_labels_list:
                    print("Warning: finetune dataset has no labels. Skipping threshold fine-tuning.")
                else:
                    best_threshold, threshold_results = predictor.find_optimal_threshold(
                        ft_proba_list,
                        ft_labels_list,
                        metric=ft_conf.get('metric', 'f1'),
                        threshold_range=tuple(ft_conf.get('threshold_range', [0.05, 0.95])),
                        threshold_step=ft_conf.get('threshold_step', 0.05),
                    )
                    print(f"Optimal threshold ({ft_conf.get('metric', 'f1')}): {best_threshold:.4f}  "
                          f"(was: {predictor.threshold})")
                    predictor.threshold = best_threshold
                    if ft_conf.get('save_search_results', True):
                        os.makedirs(str(output_dir.absolute()), exist_ok=True)
                        threshold_results.to_csv(
                            os.path.join(str(output_dir.absolute()), 'threshold_search.csv'),
                            index=False,
                        )
                        print(f"Threshold search results saved to {output_dir}/threshold_search.csv")
                # Restore the original test dataloader
                predictor.load_data(testDataLoader)
                print("="*60 + "\n")
        # ─────────────────────────────────────────────────────────────────────

        if config['ensemble_predictions']:
            results = predictor.evaluate_model_ensemble(models_list=models_list, model_checkpoints=checkpoints,
                                                        save_tiles=save_tiles, output_path=str(output_dir.absolute()),
                                                        predict_only=False, device=DEVICE,
                                                        TTA=config['tta'], num_augs=config['num_augmentations'],
                                                        dataset_csv=dataset_path.absolute(),
                                                        dinov3_config=dinov3_conf if is_dinov3_run else None
                                                        )
        else:
            if not predictor.model_is_loaded:
                predictor.load_model(model, str(checkpoint.absolute()),
                                     device=DEVICE,
                                     autoselect_gpu=config['autoselect_gpu']
                                    )
            results = predictor.evaluate_model(save_tiles=save_tiles, output_path=str(output_dir.absolute()),
                                               predict_only=False, TTA=config['tta'], num_augs=config['num_augmentations'],
                                               dataset_csv=dataset_path.absolute())
        
        if config['reconstruct_map']:
            if config['use_masked']:
                try:
                    with open(config['dataset_json'], 'r') as fin:
                        params = json.load(fin)
                    try:
                        original_image_size = params['original_input_size']
                        original_image_size = original_image_size if original_image_size !=-1 else None
                    except Exception as Error:
                        original_image_size = None
                        print("Error dealing with original_image_size! original_image_size has been set to be None")
                    # output_path_ = output_dir.parent.absolute()
                    output_map_filename=str(output_path_/'reconstructed.png')
                    ImageTiler.reconstruct_image(tile_folder_path=str(output_dir.absolute()), 
                                                output_filename=output_map_filename,
                                                target_size=original_image_size,
                                                colourize=config['colourize'])
                except Exception as Error:
                    print("Error during map reconstruction! Error type:", Error)                    
                    sys.exit(2)
            else:
                print("Error! Tried to reconstruct slum map with \'use_mask\' set to False.\n Re-run evaluation with \'use_mask\' set to True.")
                sys.exit(3)

    else:
        assert os.path.isdir(os.path.join(config['dataset_path'])), f"\nError! Supplied tile location ({config['dataset_path']}) is not a directory.""\
                                                    Inference on overlaped tiles requires the path to the tiles folder, rather than a dataset.csv file."
        im_0 = io.imread(os.path.join(config['dataset_path'], os.listdir(config['dataset_path'])[0]))
        tile_size = im_0.shape[:2] if len(im_0.shape) > 2 else im_0.shape
        # output_path_ = os.path.dirname(os.path.abspath(output_dir))
        output_filename = 'reconstructed_overlap.png'
        transforms = 'tta_'+config['image_type'] if config['tta'] else 'inference_'+config['image_type']
        predictor.get_data_from_folder(config['dataset_path'],
                                       normalization_file=config['normalization_file_path'],
                                       tile_size=tile_size,
                                       transforms=transforms,
                                       batch_size=config['batch_size'])
        predictor.check_dataloader()
        if not save_tiles:
            print("Warning! Evaluation with a dataset tiled with overlap requires tiles to be saved for the reconstruction of the full slum map.")
            print("Parameter save_tiles will thus be ignored.")
            save_tiles = True
        if config['ensemble_predictions']:
            output_dir = predictor.run_inference_with_ensemble(models_list=models_list, model_checkpoints=checkpoints,
                                                               save_tiles=save_tiles, output_path=str(output_dir.absolute()), 
                                                               device=DEVICE, TTA=config['tta'], num_augs=config['num_augmentations'])
        else:
            predictor.load_model(model, str(checkpoint.absolute()),
                                 device=DEVICE,
                                 autoselect_gpu=config['autoselect_gpu'],) 

            output_dir = predictor.run_inference(output_path=output_dir, save_tiles=save_tiles, TTA=config['tta'], num_augs=config['num_augmentations'])
        
        OverlapTiler.reconstruct_image(tiling_info_json=config['dataset_json'], 
                                       tile_folder=output_dir,
                                       output_path= output_path_, 
                                       output_filename=output_filename,
                                       make_visible=config['colourize'])

        output_map_filename=os.path.join(output_path_, output_filename)
        print("Reconstructed map file:", output_map_filename)
        results = predictor.evaluate_from_full_maps(input_slum_map=config['raw_slum_map_path'], 
                                                    predicted_slum_map=output_map_filename,
                                                    output_path=os.path.abspath(output_dir)) 
    print(results)

    if config['overlay_map'] is True:
        if config['reconstruct_map']:
            try:
                output_file_overlay = os.path.join(output_path_, "full_map_overlay.png")
                overlay(satellite_img_file=config['raw_satellite_image_path'], 
                        pred_slums_img_file=output_map_filename, 
                        output_file=output_file_overlay, 
                        mask_file=config['raw_mask_image_path'], 
                        true_slums_img_file=config['raw_slum_map_path'])
            except Exception as Error:
                print("Error during overlay operation! Error log:", Error)
                sys.exit(4)
        else:
            print("Error! overlay set to true with reconstruct set to false. You have to reconstuct the map first")
            print("Re-run evaluation setting both reconstuct and overlay to true.")
            sys.exit(5)

    if config['generate_shapefile']:
        if not config['reconstruct_map']:
            print("Error. In order to generate shapefiles you have to set the reconstuct map to True and supply the auxilliary file location.")
            print("Aborting operation...")
            sys.exit(6)
        assert os.path.exists(config['auxilliary_files_folder']), f"Could not find auxilliary_files_folder {config['auxilliary_files_folder']}"
        
        if 'epsg_code' not in config:
            warnings.warn("'epsg_code' not found in config. Falling back to EPSG:32634 (WGS 84 / UTM Zone 34N). Shapefiles may have incorrect CRS.", UserWarning)
        generate_shapefiles(input_image_path=config['raw_satellite_image_path'],
                            auxilliary_files_folder=config['auxilliary_files_folder'],
                            output_folder=output_dir,
                            shapefile_name=config['shapefile_name'],
                            reconstructed_map_file=output_map_filename,
                            crop=config['crop'],
                            produce_png_overlay=False,
                            epsg_code=config.get('epsg_code', 32634)
                            )

    if not config['use_masked']:
        # tidy up:
        os.remove(tmp_dataset_csv)
        os.rmdir(tmp_masked_folder)

    cnn_results_dir = output_dir/'nn_resuts'
    cnn_tiles_dir = output_dir/'nn_tiles'
    os.makedirs(cnn_results_dir, exist_ok=True)
    os.makedirs(cnn_tiles_dir, exist_ok=True)
    shutil.copyfile(config_file, os.path.join(cnn_results_dir, os.path.basename(config_file)))
    if config['reconstruct_map']:
        # move generated map files inside cnn_result folder
        shutil.move(output_map_filename, os.path.join(cnn_results_dir, os.path.basename(output_map_filename)))
        shutil.move(output_file_overlay, os.path.join(cnn_results_dir, os.path.basename(output_file_overlay)))
    for file_i in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, file_i)):
            if not file_i.endswith('.png'):
                # move other results to cnn_results folder
                shutil.move(os.path.join(output_dir, file_i), os.path.join(cnn_results_dir, file_i))
            else:
                # move tiles to cnn_tiles folder
                shutil.move(os.path.join(output_dir, file_i), os.path.join(cnn_tiles_dir, file_i))
    # tify up
    if os.path.exists(output_path_) and os.path.isdir(output_path_):
        shutil.rmtree(output_path_)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-c', '--config_file', type=str, default=Path.home()/'slumworldML'/'config'/'evaluate.yml', help="Configuration yml file for running the evaluation script. Should point to the"\
        " saved checkpoint file, the datapath, the normalization file and may include additional flags. Default: \'~\/slumworldML\/config\/evaluate.yml\'")
    args = vars(parser.parse_args())
    try:
        with open(args['config_file'], 'r') as fin:
            config = yaml.safe_load(fin)
            print(config)
    except Exception as Error:
        print("Error! Could not load configuration file. Reason:", Error)
        sys.exit(1)

    main(config, config_file=args['config_file'])