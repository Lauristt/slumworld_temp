"""
Last updated: 2026-04-05 by Yuting
This code now supports absolute counts for the split ratio.

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


def parse_split_argument(split_values, total_samples):
    """
    Parse split argument and determine if it's fraction or absolute count mode.
    
    Args:
        split_values (list): List of 3 floats/ints representing [Train, Val, Test]
        total_samples (int): Total number of samples in dataset
        
    Returns:
        tuple: (n_train, n_val, n_test, mode_str)
        
    Raises:
        ValueError: If split values are invalid
    """
    # Determine mode based on values
    # If all values are <= 1.0 and sum to ~1.0, treat as fractions
    # If any value is > 1.0, treat as absolute counts
    
    all_fractions = all(v <= 1.0 for v in split_values)
    any_integers = any(v >= 1.0 for v in split_values)
    
    # Check consistency
    has_decimals = any(v % 1 != 0 for v in split_values)
    
    if all_fractions and not any_integers:
        # Fraction mode
        mode = "fraction"
        total = sum(split_values)
        if abs(total - 1.0) > 1e-6:  # Allow small floating point error
            raise ValueError(f"Split fractions must sum to 1.0, got {total}. Input: {split_values}")
        
        n_train = int(total_samples * split_values[0])
        n_val = int(total_samples * split_values[1])
        n_test = total_samples - n_train - n_val  # Remaining samples to handle rounding
        
        print(f"  Mode: Fraction-based | {split_values[0]:.1%} Train, {split_values[1]:.1%} Val, {split_values[2]:.1%} Test")
        
    elif any_integers and not (all_fractions and has_decimals):
        # Absolute count mode
        mode = "absolute"
        n_train, n_val, n_test = [int(v) for v in split_values]
        
        total_specified = n_train + n_val + n_test
        if total_specified != total_samples:
            raise ValueError(
                f"Split counts must sum to total samples ({total_samples}), "
                f"got {total_specified} (Train={n_train}, Val={n_val}, Test={n_test})"
            )
        
        print(f"  Mode: Absolute count | Train={n_train}, Val={n_val}, Test={n_test}")
    else:
        raise ValueError(
            f"Invalid split format. Use either:\n"
            f"  - All fractions ≤ 1.0 summing to 1.0 (e.g., 0.7 0.2 0.1)\n"
            f"  - Absolute counts ≥ 1 summing to total samples (e.g., 70 20 10)\n"
            f"  Got: {split_values}"
        )
    
    return n_train, n_val, n_test, mode


def process_custom_dataset(input_dir, output_dir, binarize=True, split_ratio=[1.0, 0.0, 0.0]):
    """
    Core function to process the dataset.

    Args:
        input_dir (str): Path to the folder containing 'image' and 'label' subfolders.
        output_dir (str): Path where the processed dataset will be saved.
        binarize (bool): Whether to force label binarization (0/1). Default: True.
        split_ratio (list): A list of 3 values representing [Train, Validation, Test].
                            Can be either:
                            - Fractions (all ≤ 1.0, summing to 1.0): [0.7, 0.15, 0.15]
                            - Absolute counts (integers ≥ 1): [70, 20, 10]
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    img_src = input_path / 'image'
    lbl_src = input_path / 'label'

    if not img_src.exists() or not lbl_src.exists():
        raise FileNotFoundError(f"Error: Input directory '{input_path}' must contain 'image' and 'label' subfolders.")


    tiled_input = output_path / 'tiled_input'
    tiled_labels = output_path / 'tiled_labels'
    tiled_input.mkdir(parents=True, exist_ok=True)
    tiled_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in os.listdir(img_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    records = []

    print(f"Found {len(image_files)} images. Processing...")

    for fname in image_files:
        src_img = os.path.join(img_src, fname)

        #match filename
        stem = Path(fname).stem
        if "_input" in stem:
            lbl_name = stem.replace("_input", "_label") + ".png"
        else:
            lbl_name = fname

        src_lbl = os.path.join(lbl_src, lbl_name)

        if os.path.exists(src_lbl):
            dst_img = tiled_input / fname
            dst_lbl = tiled_labels / lbl_name

            shutil.copy2(src_img, dst_img)

            label_arr = io.imread(src_lbl)
            if binarize:
                label_arr = CNNTiler.convert_y_labels_to_binary(label_arr)
            
            # save label
            io.imsave(dst_lbl, label_arr, check_contrast=False)

            # calculate static coverage
            coverage = np.count_nonzero(label_arr) / label_arr.size

            # create record
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

    # generate dataframe
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Error: No valid image-label pairs found. Please check filenames.")

    # apply data splitting logic
    if split_ratio != [1.0, 0.0, 0.0]:
        print(f"Applying dataset split ratio: {split_ratio}")
        
        try:
            n_train, n_val, n_test, mode = parse_split_argument(split_ratio, len(df))
        except ValueError as e:
            print(f"Error parsing split argument: {e}")
            sys.exit(1)
        
        # randomly shuffle all indices to ensure random distribution
        permuted_indices = np.random.permutation(len(df))
        
        # assign indices to specific sets
        train_idx = permuted_indices[:n_train]
        val_idx = permuted_indices[n_train : n_train + n_val]
        test_idx = permuted_indices[n_train + n_val : n_train + n_val + n_test]
        
        # apply assignment (Train is already default)
        if len(val_idx) > 0:
            df.loc[val_idx, 'dataset_part'] = 'Validation'
        if len(test_idx) > 0:
            df.loc[test_idx, 'dataset_part'] = 'Test'
        
        print(f"  Split results: Train={len(train_idx)}, Validation={len(val_idx)}, Test={len(test_idx)}")

    # calculate dataset statistics (mean/std)
    print("Calculating Dataset Mean/Std...")
    try:
        # uses CNNTiler to compute channel-wise mean and std for normalization
        mean, std, n_channels = CNNTiler.calculate_mean_std(df)
    except Exception as e:
        print(f"Statistics calculation failed: {e}. Using default values.")
        mean, std, n_channels = [0.0]*3, [1.0]*3, 3

    # save metadata: dataset.json
    summary = {
        "mean": mean,
        "std": std,
        "num_of_channels": n_channels,
        "num_of_tiles": len(df),
        "original_input_size": -1  # -1 indicates manual/custom sizing
    }
    with open(output_path / 'dataset.json', 'w') as jf:
        json.dump(summary, jf, indent=4)

    # save metadata: dataset.csv
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
                        help="Split parameter: [Train, Validation, Test]. Can be either:\n"
                             "  (1) Fractions (≤ 1.0, sum to 1.0): -s 0.7 0.15 0.15\n"
                             "  (2) Absolute counts (≥ 1): -s 70 20 10\n"
                             "  Default: [1.0, 0.0, 0.0] (All Train)")

    args = parser.parse_args()

    try:
        process_custom_dataset(args.input_dir, args.output_dir, args.binarize, args.split)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()