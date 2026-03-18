"""
finetune_dummy_tiler.py

A specialized script to process manually selected (pre-tiled) images and labels 
into the standard dataset format required by the SatelliteDataModule.

Key Features:
1. Matches image and label pairs (handles '_input' vs '_label' suffixes).
2. Copies images and processes labels (optional binarization).
3. Auto-fills "perfect" statistics (slum_sampling_prob=1.0) to prevent data filtering.
4. Supports Train/Validation/Test splitting.
5. Calculates dataset mean/std and generates 'dataset.csv' and 'dataset.json'.
"""

import os
import sys
import shutil
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from skimage import io
import warnings

# Suppress low contrast warnings when saving binary masks
warnings.filterwarnings("ignore")

# Attempt to import the CNNTiler for statistical utilities
try:
    from slumworldML.src.cnn_tiler import CNNTiler
except ImportError:
    try:
        from src.cnn_tiler import CNNTiler
    except ImportError:
        # Fallback for flat directory structures
        from cnn_tiler import CNNTiler


def process_custom_dataset(input_dir, output_dir, binarize=True, split_ratio=[1.0, 0.0, 0.0]):
    """
    Core function to process the dataset.

    Args:
        input_dir (str): Path to the folder containing 'image' and 'label' subfolders.
        output_dir (str): Path where the processed dataset will be saved.
        binarize (bool): Whether to force label binarization (0/1). Default: True.
        split_ratio (list): A list of 3 floats representing [Train, Validation, Test] fractions.
                            Must sum to 1.0 (e.g., [0.7, 0.15, 0.15]).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    img_src = input_path / 'image'
    lbl_src = input_path / 'label'

    # 1. Validation: Ensure source folders exist
    if not img_src.exists() or not lbl_src.exists():
        raise FileNotFoundError(f"Error: Input directory '{input_path}' must contain 'image' and 'label' subfolders.")

    # 2. Setup Output Directory Structure
    tiled_input = output_path / 'tiled_input'
    tiled_labels = output_path / 'tiled_labels'
    tiled_input.mkdir(parents=True, exist_ok=True)
    tiled_labels.mkdir(parents=True, exist_ok=True)

    # 3. File Matching and Processing
    image_files = sorted([f for f in os.listdir(img_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    records = []

    print(f"Found {len(image_files)} images. Processing...")

    for fname in image_files:
        src_img = os.path.join(img_src, fname)

        # --- Filename Matching Logic ---
        # Handles cases like: "area1_input.png" -> "area1_label.png"
        stem = Path(fname).stem
        if "_input" in stem:
            lbl_name = stem.replace("_input", "_label") + ".png"
        else:
            # Fallback: assume label has the exact same name
            lbl_name = fname

        src_lbl = os.path.join(lbl_src, lbl_name)

        if os.path.exists(src_lbl):
            dst_img = tiled_input / fname
            dst_lbl = tiled_labels / lbl_name

            # A. Process Image: Direct Copy
            shutil.copy2(src_img, dst_img)

            # B. Process Label: Read -> (Optional Binarize) -> Save
            label_arr = io.imread(src_lbl)
            if binarize:
                # Use standard logic: Values >= 64 become 1, others 0
                label_arr = CNNTiler.convert_y_labels_to_binary(label_arr)
            
            # Save label (check_contrast=False prevents warnings for all-black/white images)
            io.imsave(dst_lbl, label_arr, check_contrast=False)

            # C. Calculate Static Coverage
            coverage = np.count_nonzero(label_arr) / label_arr.size

            # D. Create Record
            # We initially set 'dataset_part' to 'Train'. This will be updated later if splitting is requested.
            # We strictly set 'slum_sampling_prob' to 1.0 to ensure these tiles are not filtered out during training.
            records.append({
                "x_location": str(dst_img.absolute()),
                "y_location": str(dst_lbl.absolute()),
                "dataset_part": "Train",
                "slum_coverage": coverage,
                "slum_sampling_prob": 1.0, 
                "average_slum_coverage": coverage,
                "min_coverage": coverage,
                "max_coverage": coverage
            })
        else:
            print(f"Warning: Label match not found for {fname}. Skipping.")

    # 4. Generate DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Error: No valid image-label pairs found. Please check filenames.")

    # 5. Apply Data Splitting Logic
    if split_ratio != [1.0, 0.0, 0.0]:
        print(f"Applying dataset split ratio: {split_ratio}")
        
        # Randomly shuffle all indices to ensure random distribution
        permuted_indices = np.random.permutation(len(df))
        
        # Calculate split boundaries
        n_train = int(len(df) * split_ratio[0])
        n_val = int(len(df) * split_ratio[1])
        
        # Assign indices to specific sets
        # Note: 'Train' is already the default, so we only need to update Validation and Test
        val_idx = permuted_indices[n_train : n_train + n_val]
        test_idx = permuted_indices[n_train + n_val :]
        
        df.loc[val_idx, 'dataset_part'] = 'Validation'
        df.loc[test_idx, 'dataset_part'] = 'Test'
        
        print(f"Split results: Train={n_train}, Validation={len(val_idx)}, Test={len(test_idx)}")

    # 6. Calculate Dataset Statistics (Mean/Std)
    print("Calculating Dataset Mean/Std...")
    try:
        # Uses CNNTiler to compute channel-wise mean and std for normalization
        mean, std, n_channels = CNNTiler.calculate_mean_std(df)
    except Exception as e:
        print(f"Statistics calculation failed: {e}. Using default values.")
        mean, std, n_channels = [0.0]*3, [1.0]*3, 3

    # 7. Save Metadata: dataset.json
    summary = {
        "mean": mean,
        "std": std,
        "num_of_channels": n_channels,
        "num_of_tiles": len(df),
        "original_input_size": -1  # -1 indicates manual/custom sizing
    }
    with open(output_path / 'dataset.json', 'w') as jf:
        json.dump(summary, jf, indent=4)

    # 8. Save Metadata: dataset.csv
    csv_path = output_path / 'dataset.csv'
    df.to_csv(csv_path, index=False)

    print(f"Done. Dataset successfully generated at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process manual tiles into standard dataset format with optional splitting.")
    
    parser.add_argument('-i', '--input_dir', type=str, required=True, 
                        help="Input folder containing 'image' and 'label' subfolders.")
    
    parser.add_argument('-o', '--output_dir', type=str, required=True, 
                        help="Output folder where tiled_input, tiled_labels, csv and json will be saved.")
    
    parser.add_argument('--binarize', action='store_true', default=True,
                        help="Convert labels to 0/1 (binary) regardless of input value. Default: True")
    
    parser.add_argument('-s', '--split', nargs=3, type=float, default=[1.0, 0.0, 0.0],
                        help="Train, Validation, Test split fractions (must sum to 1.0). "
                             "Example: -s 0.7 0.15 0.15. Default: [1.0, 0.0, 0.0] (All Train)")

    args = parser.parse_args()

    try:
        process_custom_dataset(args.input_dir, args.output_dir, args.binarize, args.split)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()