''' Load a distance from boundary slums label file, binarise it and save it to disk.
'''

from PIL import Image
import numpy as np
import click

Image.MAX_IMAGE_PIXELS = None

def load_and_binarize_image(image_path, threshold=64):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert image to grayscale (monochrome)
    image_array = np.array(image)
    min_val = image_array.min()
    max_val = image_array.max()
    print(f"image min: {image_array.min()}")
    print(f"image max: {image_array.max()}")
    num_pixels = image_array.size
    num_below_threshold = np.sum(image_array < threshold)
    percentage_below_threshold = (num_below_threshold / num_pixels) * 100
    print(f"Percentage of pixels < {threshold}: {percentage_below_threshold:.10f}%")

    # Binarize the image
    
    binarized_image = (image_array >= threshold).astype(np.uint8)  # Thresholding to create binary image
    return binarized_image

def save_binarized_image(binarized_image, output_path):
    # Convert binary image to PIL image and save
    binarized_pil_image = Image.fromarray(binarized_image * 255)  # Convert binary image back to PIL image
    print(f"image min: {np.array(binarized_pil_image).min()}")
    print(f"image max: {np.array(binarized_pil_image).max()}")
    binarized_pil_image.save(output_path)
    print(f"Binarized image saved to {output_path}")


@click.command()
@click.option('--image_path', required=True, help='Path to the input image (eg, "/mnt/24B4E926B4E8FAE6/slumworld/data/raw/Mumbai/2001/PAN/version007/MS/image_dir/input_y.png")')
@click.option('--output_path', required=True, help='Path to save the binarized image (eg "mnt/24B4E926B4E8FAE6/slumworld/data/raw/Mumbai/2001/PAN/version007/MS/image_dir/input_y_binarised.png")')
@click.option('--threshold', default=64, help='Threshold for binarization', show_default=True)
@click.help_option('-h', '--help')

def main(image_path, output_path, threshold):
    # Load and binarize the image
    binarized_image = load_and_binarize_image(image_path, threshold)
    
    # Save the binarized image
    save_binarized_image(binarized_image, output_path)

if __name__ == "__main__":
    main()
