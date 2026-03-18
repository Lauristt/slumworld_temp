''' Helper function for inspection of the alignment of two shapefiles and basic union/intersection/symmetric difference operations.
Load two shapefiles, convert them to image, visualise, compute intersection/union/symmetric difference, save and plot to screen'''

import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import click

Image.MAX_IMAGE_PIXELS = None

def remove_large_polygon(gdf):
    """
    Remove a polygon that covers the entire extent of the GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame containing polygons.

    Returns:
    GeoDataFrame: Cleaned GeoDataFrame with the large polygon removed, if it existed.
    """
    # Get the total extent of the GeoDataFrame
    total_bounds = gdf.total_bounds
    total_extent = (total_bounds[0], total_bounds[1], total_bounds[2], total_bounds[3])
    
    # Function to check if a polygon covers the total extent
    def is_large_polygon(geometry, total_extent):
        bounds = geometry.bounds
        return bounds == total_extent

    # Find the indices of polygons that cover the total extent
    large_polygon_indices = gdf[gdf.geometry.apply(lambda x: is_large_polygon(x, total_extent))].index
    
    # Remove these polygons from the GeoDataFrame
    if not large_polygon_indices.empty:
        gdf_cleaned = gdf.drop(large_polygon_indices)
    else:
        gdf_cleaned = gdf
    
    return gdf_cleaned

@click.command()
@click.option('--shapefile1_path', required=True, help='Path to the first shapefile')
@click.option('--shapefile2_path', required=True, help='Path to the second shapefile')
@click.option('--pixels_per_meter', default=1, help='Resolution in pixels per meter', show_default=True)
@click.help_option('-h', '--help')

def main(shapefile1_path, shapefile2_path, pixels_per_meter):
    # Load the shapefiles
    gdf1 = remove_large_polygon(gpd.read_file(shapefile1_path))
    gdf2 = remove_large_polygon(gpd.read_file(shapefile2_path))

    # Determine the bounding box
    bbox1 = gdf1.total_bounds
    bbox2 = gdf2.total_bounds

    # Calculate the union of the bounding boxes
    min_x = min(bbox1[0], bbox2[0])
    min_y = min(bbox1[1], bbox2[1])
    max_x = max(bbox1[2], bbox2[2])
    max_y = max(bbox1[3], bbox2[3])
    bounding_box = (min_x, min_y, max_x, max_y)

    # Calculate the intersection of the bounding boxes
    min_x = max(bbox1[0], bbox2[0])
    min_y = max(bbox1[1], bbox2[1])
    max_x = min(bbox1[2], bbox2[2])
    max_y = min(bbox1[3], bbox2[3])
    intersection_bounding_box = (min_x, min_y, max_x, max_y)

    # Calculate the width and height in meters
    width_meters = max_x - min_x
    height_meters = max_y - min_y

    # Calculate the image dimensions in pixels
    width_pixels = int(width_meters * pixels_per_meter)
    height_pixels = int(height_meters * pixels_per_meter)

    def plot_and_save_aligned(gdf, bounding_box, output_path, width_pixels, height_pixels):
        fig, ax = plt.subplots(figsize=(width_pixels / 100, height_pixels / 100), dpi=100)
        gdf.plot(ax=ax)
        ax.set_xlim([bounding_box[0], bounding_box[2]])
        ax.set_ylim([bounding_box[1], bounding_box[3]])
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

    plot_and_save_aligned(gdf1, intersection_bounding_box, 'shapefile1.png', width_pixels, height_pixels)
    plot_and_save_aligned(gdf2, intersection_bounding_box, 'shapefile2.png', width_pixels, height_pixels)

    def load_and_binarize_image(image_path, threshold=128):
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        binary_image = (np.array(image) < threshold).astype(np.uint8) * 255
        binary_image_pil = Image.fromarray(binary_image)
        return binary_image_pil, binary_image

    binary_image1_pil, binary_image1_np = load_and_binarize_image('shapefile1.png')
    binary_image2_pil, binary_image2_np = load_and_binarize_image('shapefile2.png')

    # Save the binarized images
    binary_image1_pil.save('shapefile1.png')
    binary_image2_pil.save('shapefile2.png')

    def load_png_as_numpy(image_path):
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        return np.array(image)

    image1_np = load_png_as_numpy('shapefile1.png')
    image2_np = load_png_as_numpy('shapefile2.png')

    # Union
    union_np = np.maximum(image1_np, image2_np)
    print(f"union shape: {union_np.shape}")
    print(f"union pixels: {np.sum(union_np)}")

    # Intersection
    intersection_np = np.minimum(image1_np, image2_np)
    print(f"intersection shape: {intersection_np.shape}")
    print(f"intersection pixels: {np.sum(intersection_np)}")

    # Symmetric difference
    symmetric_difference = np.logical_xor(image1_np, image2_np).astype(np.uint8) * 255
    print(f"symmetric_difference shape: {symmetric_difference.shape}")
    print(f"symmetric_difference pixels: {np.sum(symmetric_difference)}")

    # Save the resulting images for visualization if needed
    Image.fromarray(union_np).save('union.png')
    Image.fromarray(intersection_np).save('intersection.png')
    Image.fromarray(symmetric_difference).save('symmetric_difference.png')

if __name__ == "__main__":
    main()
