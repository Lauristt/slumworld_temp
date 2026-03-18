import sys
import os
from pathlib import Path
import argparse
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

try:
    from slumworldML.src.cnn_tiler import CNNTiler
    from slumworldML.src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
    from slumworldML.src.custom_transformations import BinarizeLabels
except ImportError:
    try:
        from src.cnn_tiler import CNNTiler
        from src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
        from src.custom_transformations import BinarizeLabels
    except ImportError as e:
        raise RuntimeError("Failed to import required modules") from e

def run(args):
    base_input_dir = PROJECT_ROOT / "raw" / "Bobo" / "Diff_Brightness"
    output_root = PROJECT_ROOT / "tiled" / "Bobo" / "tiled_input"
    
    gamma_values = ['0.2', '0.5', '0.8']  

    for gamma in gamma_values:

        x_path = base_input_dir / "input_x" / f"input_x_enhanced_gamma{gamma}.png"
        y_path = base_input_dir / "input_y" / f"input_y_enhanced_gamma{gamma}.png"
        z_path = base_input_dir / "input_z" / f"input_z_enhanced_gamma{gamma}.png"

  
        if not x_path.exists():
            print(f"⚠️ Skipping gamma {gamma}: X file missing at {x_path}")
            continue

        output_dir = output_root / f"Diff_Brightness_gamma{gamma}"
        output_dir.mkdir(parents=True, exist_ok=True)

        tiler = CNNTiler(
            tile_size=args.tile_size,
            save_path=output_dir
        )

        print(f"\n{'='*30}")
        print(f"Processing gamma {gamma}")
        print(f"Input X: {x_path}")
        print(f"Output directory: {output_dir}")

        tiler.create_standard_dataset(x_input_path=x_path,
                                        y_input_path=y_path, 
                                        training_split=args.split,
                                        mask=z_path,
                                        labels2binary=args.labels2binary)
    
        if y_path.exists() and args.split[0] > 0:
            print("Generating dataset statistics...")
            transformation = TRAINING_TRANSFORMS_BASIC
            if not args.labels2binary:
                transformation['joint_transforms'].insert(0, BinarizeLabels())
            transformation = create_transform(TRAINING_TRANSFORMS_BASIC, mean=[0,0,0], std=[1,1,1]) 
            tiler.calculate_tile_statistics(
                transformation=transformation,
                num_of_samples=100,
                dataset_name=f"dataset.csv",
                save_csv=True
            )

def validate_args(args):
    if not (0.999 <= sum(args.split) <= 1.001):
        raise ValueError(f"Invalid split sum: {sum(args.split):.3f}, must sum to 1.0")
    
    if any(not 0 <= x <= 1 for x in args.split):
        raise ValueError("All split values must be between 0 and 1")

    if args.tile_size < 64:
        raise ValueError("Tile size must be at least 64 pixels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gamma Image Tiling Processor")
    
    parser.add_argument("-t", "--tile_size", type=int, required=True,
                        help="Tile size in pixels")
    
    parser.add_argument("-s", "--split", nargs=3, type=float, default=[0.70, 0.15, 0.15], # 0.75,0.15,0.15
                        help="Train/Val/Test split ratios")
    # parser.add_argument("--overlap", type=int, default=0,
    #                     help="Overlap between tiles in pixels")
    parser.add_argument("--labels2binary", action="store_true",
                        help="Binarize label masks")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--tile_format", choices=["png", "jpg"], default="png",
                        help="Output tile format")
    parser.add_argument("-c", "--config", type=str,
                        help="Path to YAML config file")

    args = parser.parse_args()

    if args.config:
        config_file = Path(args.config).resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
            parser.set_defaults(**config)
            args = parser.parse_args()

    try:
        validate_args(args)
        run(args)
        print("\n✅ All processing completed successfully")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)