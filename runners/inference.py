''' Script for running inference with a trained model.
The script loads a trained model, runs it on a given dataset and saves the results.
Optionally it can also reconstruct the original map (if run on all image tiles).
It reads all required parameteres (experiment_path , checkpoint_path, data_path, 
norm_file_path, reconstruct_map, output_dir: ) from a configuration.yml file.
The only argument that needs to be passed to the script is the location of the configuration file.
ARGS:
    -c [--config_file]:             str, full path to the configuration file [default: \'~\/home\/user\/slumworldML\/config\/inference.yml\']
USAGE:
    >>> python3 inference.py -c ../config/inference.yml
'''
import sys
import shutil
import os
import shutil
import pdb
from torchvision.transforms.functional import convert_image_dtype
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
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

    dataset_path = Path(config['dataset_path'])
    norm_file = Path(config['normalization_file_path'])
    save_tiles = config['save_tiles']
    num_channels = 3 if config['image_type'] == "mul" else 1
    num_channels = 3 # verify if we need to have the 'pan' single channel option

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
        model = MODELS_REGISTRY[model_i](in_channels=num_channels, out_channels=1, domain_classifier=False)
    DEVICE = config['device']   

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

        Loader = TestingDataLoader(str(dataset_path.absolute()), batch_size=config['batch_size'], 
                                norm_file=str(norm_file.absolute()), tile_size=tile_size, 
                                split_tiles=config['split_tiles'], TTA=config['tta'], 
                                image_type=config['image_type'],
                                mode='infer',
                                use_only_test_tiles=config['use_only_test_tiles'])
        testDataLoader = Loader.get_dataloader()

        predictor.load_data(testDataLoader)

        if config['ensemble_predictions']:
            _ = predictor.run_inference_with_ensemble(models_list=models_list, model_checkpoints=checkpoints,
                                                            save_tiles=save_tiles, output_path=str(output_dir.absolute()), 
                                                            device=DEVICE, TTA=config['tta'], num_augs=config['num_augmentations']
                                                            )
        else:
            predictor.load_model(model, str(checkpoint.absolute()),
                                 device=DEVICE, 
                                 autoselect_gpu=config['autoselect_gpu']
                                )
            results = predictor.run_inference(save_tiles=save_tiles, output_path=str(output_dir.absolute()), 
                                              TTA=config['tta'], num_augs=config['num_augmentations']
                                              )

        if config['reconstruct_map']:
            if config['use_masked']:
                try:
                    with open(config['dataset_json'], 'r') as fin:
                        params = json.load(fin)
                    original_image_size = params['original_input_size']
                    output_path_ = output_dir.parent.absolute()
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
        output_path_ = os.path.dirname(os.path.abspath(output_dir))
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
        
        generate_shapefiles(input_image_path=config['raw_satellite_image_path'],
                            auxilliary_files_folder=config['auxilliary_files_folder'],
                            output_folder=output_path_,
                            shapefile_name=config['shapefile_name'],
                            reconstructed_map_file=output_map_filename,
                            crop=config['crop'],
                            produce_png_overlay=False
                            )

    if not config['use_masked']:
        # tidy up:
        os.remove(tmp_dataset_csv)
        os.rmdir(tmp_masked_folder)
    
    shutil.copyfile(config_file, os.path.join(output_dir, os.path.basename(config_file)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-c', '--config_file', type=str, default=Path.home()/'slumworldML'/'config'/'inference.yml', help="Configuration yml file for running the inference script. Should point to the"\
        " saved checkpoint file, the datapath, the normalization file and may include additional flags. Default: \'~\/slumworldML\/config\/inference.yml\'")
    args = vars(parser.parse_args())
    try:
        with open(args['config_file'], 'r') as fin:
            config = yaml.safe_load(fin)
            print(config)
    except Exception as Error:
        print("Error! Could not load configuration file. Reason:", Error)
        sys.exit(1)

    main(config, config_file=args['config_file'])