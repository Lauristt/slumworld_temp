import argparse
import os
import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import requests
from torchvision.io import read_image
from torchvision import transforms
from tqdm import tqdm

def load_dino_feature_extractor(repo_path, model_config,device='cuda',cache_dir = None):
    print("Loading special DINOv3 satellite-trained model for feature extraction...")
    model_url = model_config['url']
    hub_model_name = model_config['hub_model_name']
    file_name = model_url.split('/')[-1].split('?')[0]
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/hub/checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, file_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading satellite model to {model_path}...")
        with requests.get(model_url, stream=True) as r, open(model_path, "wb") as f, tqdm(
            unit="B", unit_scale=True, unit_divisor=1024, 
            total=int(r.headers.get('content-length', 0)), desc=file_name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Download complete.")
    
    if not os.path.exists(repo_path):
         raise FileNotFoundError(f"DINOv3 local repo not found at {repo_path}. Please provide the correct path.")
    
    model = torch.hub.load(repo_path, hub_model_name, source='local', weights=model_path)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    model.to(device)
    print("DINOv3 feature extractor loaded successfully to device:", device)
    return model


def main(args):
    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    print(f"Using device for preprocessing: {device}")
    model_config = json.loads(args.model_config_json)

    dino_model = load_dino_feature_extractor(args.dino_repo_path, model_config,device,args.cache_dir)
    dino_feature_dim = model_config['feature_dim']

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Features will be saved to: {args.output_dir}")

    df = pd.read_csv(args.csv_path)
    if 'x_location' not in df.columns:
        raise ValueError("CSV file must contain an 'x_location' column.")
    image_paths = df['x_location'].unique().tolist()
    print(f"Found {len(image_paths)} unique images to process.")

    with open(args.norm_file, 'r') as f:
        norm_stats = json.load(f)
    transform = transforms.Normalize(mean=norm_stats['mean'], std=norm_stats['std'])

    for image_path in tqdm(image_paths, desc="Preprocessing DINOv3 Features"):
        try:
            base_name = os.path.basename(image_path)
            output_filename = base_name.replace('.png', '.pt').replace('.jpg', '.pt')
            output_path = os.path.join(args.output_dir, output_filename)

            if os.path.exists(output_path) and not args.force_rerun:
                continue
            
            image_tensor = read_image(image_path)
            if image_tensor.shape[0] == 4:
                image_tensor = image_tensor[:3]

            image_tensor_float = image_tensor.float() / 255.0
            image_tensor_normalized = transform(image_tensor_float)
            
            image_for_dino = image_tensor_normalized.unsqueeze(0).to(device)
            if image_for_dino.shape[1] != 3:
                image_for_dino = image_for_dino.repeat(1, 3, 1, 1)
            # image_for_dino = F.interpolate(image_for_dino, size=(224, 224), mode='bicubic', align_corners=False)

            with torch.no_grad():
                patch_tokens = dino_model.get_intermediate_layers(image_for_dino, n=1)[0]
                # no cls token for dinov3
                num_patches = patch_tokens.shape[1]
                h = w = int(num_patches ** 0.5)
                if h * w != num_patches:
                    patch_size = dino_model.patch_embed.patch_size[0]
                    h = image_for_dino.shape[2] // patch_size
                    w = image_for_dino.shape[3] // patch_size
                feature_map = patch_tokens.permute(0, 2, 1).view(-1, dino_feature_dim, h, w)
            torch.save(feature_map.squeeze(0).cpu(), output_path)

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute DINOv3 features for a dataset.")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--norm_file', type=str, required=True)
    parser.add_argument('--dino_repo_path', type=str, required=True)
    parser.add_argument('--model_config_json', type=str, required=True, help='JSON string of the model config from the YAML file.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Optional path to the torch hub cache directory.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--force_rerun', action='store_true', help="Force regeneration of existing feature files.")
    args = parser.parse_args()
    main(args)