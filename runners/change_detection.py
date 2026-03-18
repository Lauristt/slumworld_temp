import os
import click
import yaml
import json
import numpy as np
import geopandas as gpd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt

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

def compute_metrics(y_true, y_pred):
    iou = np.sum((y_true & y_pred)) / np.sum((y_true | y_pred))
    precision = np.sum((y_true & y_pred)) / np.sum(y_pred)
    recall = np.sum((y_true & y_pred)) / np.sum(y_true)
    f1 = 2 * (precision * recall) / (precision + recall)
    return iou, precision, recall, f1

def compute_change_metrics(y_true_t1, y_pred_t1, y_true_t2, y_pred_t2):
    change_true = np.logical_xor(y_true_t1, y_true_t2).astype(int)
    change_pred = np.logical_xor(y_pred_t1, y_pred_t2).astype(int)
    cda = np.mean(change_true == change_pred)
    intersection = np.sum((change_true & change_pred))
    union = np.sum((change_true | change_pred))
    change_iou = intersection / union if union != 0 else 0
    tprc = intersection / np.sum(change_true) if np.sum(change_true) != 0 else 0
    return cda, change_iou, tprc

def compute_temporal_stability(predictions):
    changes = np.sum(predictions[:, 1:] != predictions[:, :-1], axis=1)
    stable_pixels = np.sum(changes == 0)
    total_pixels = predictions.shape[0]
    temporal_stability = stable_pixels / total_pixels
    return temporal_stability

def compute_consistency_index(predictions):
    most_frequent_class = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
    consistent_counts = np.sum(predictions == most_frequent_class[:, np.newaxis], axis=1)
    total_time_steps = predictions.shape[1]
    consistency_index = np.mean(consistent_counts / total_time_steps)
    return consistency_index

def compute_area_metrics(y_true_t1, y_pred_t1, y_true_t2, y_pred_t2):
    slum_area_true_t1 = np.sum(y_true_t1, dtype=np.int64)
    slum_area_true_t2 = np.sum(y_true_t2, dtype=np.int64)
    tsac = slum_area_true_t2 - slum_area_true_t1
    new_slum_true = np.logical_and(y_true_t2, np.logical_not(y_true_t1)).astype(int)
    new_slum_pred = np.logical_and(y_pred_t2, np.logical_not(y_pred_t1)).astype(int)
    nsadr = np.sum(new_slum_pred & new_slum_true) / np.sum(new_slum_true) if np.sum(new_slum_true) != 0 else 0
    converted_to_non_slum_true = np.logical_and(np.logical_not(y_true_t2), y_true_t1).astype(int)
    converted_to_non_slum_pred = np.logical_and(np.logical_not(y_pred_t2), y_pred_t1).astype(int)
    fhcr = np.sum(converted_to_non_slum_pred & converted_to_non_slum_true) / np.sum(converted_to_non_slum_true) if np.sum(converted_to_non_slum_true) != 0 else 0
    return tsac, nsadr, fhcr

def load_data(file_path):
    y = np.array(Image.open(file_path))
    if len(y.shape) > 2:
        y = y[:,:,0]
    print(f"loaded image: {os.path.basename(file_path)}")
    print(f"image shape: {y.shape}")
    return (y > 0).astype(np.uint8)

def save_results(output_filename, text, iou_t1, precision_t1, recall_t1, f1_t1,
                 iou_t2, precision_t2, recall_t2, f1_t2,
                 cda, change_iou, tprc, tsac, nsadr, fhcr,
                 temporal_stability_true, consistency_index_true,
                 temporal_stability_preds, consistency_index_preds):
    results = {
        "description": text,
    "metrics": {
            "t1": {
                "IoU": float(iou_t1),
                "Precision": float(precision_t1),
                "Recall": float(recall_t1),
                "F1": float(f1_t1)
            },
            "t2": {
                "IoU": float(iou_t2),
                "Precision": float(precision_t2),
                "Recall": float(recall_t2),
                "F1": float(f1_t2)
            },
            "change_metrics": {
                "CDA": {"value":float(cda), "description":"Change Detection Accuracy (CDA): Measures the proportion of correctly detected changes (both slum to non-slum and non-slum to slum) over all possible changes."},
                "Change_IoU": {"value":float(change_iou), "description": "Change IoU: Similar to IoU but specifically for the regions where changes occurred. This can help isolate the performance of the model in detecting changes rather than static slum areas."},
                "TPRC": {"value":float(tprc), "description":"True Positive Rate of Change (TPRC): Measures the proportion of correctly detected change pixels (slum to non-slum and vice versa) out of all actual change pixels."},
            },
            "area_metrics": {
                "TSAC": {"value":float(tsac), "description":"Total Slum Area Change (TSAC): Measures the net change in slum area over time. This helps in understanding the growth or reduction of slum areas.\n"},
                "NSADR": {"value":float(nsadr), "description":"New Slum Area Detection Rate (NSADR): Measures the rate at which new slum areas are correctly detected."},
                "FHCR": {"value":float(fhcr), "description":"Formal Housing Conversion Rate (FHCR): Measures the rate at which slum areas are correctly detected as having been converted to formal housing."},
            },
            "temporal_metrics_true": {
                "Temporal_Stability": {"value":float(temporal_stability_true), "description":"Temporal Stability: Assesses how stable the true slums (based on the labels) are over time."},
                "Consistency_Index": {"value":float(consistency_index_true), "description":"Consistency Index: Measures the consistency of slum/non-slum areas over multiple time steps."},
            },
            "temporal_metrics_preds": {
                "Temporal_Stability": {"value":float(temporal_stability_preds),"description":"Temporal Stability: Assesses how stable the model's predictions are over time, which is crucial for change detection tasks. Frequent false positives/negatives over time can indicate instability."},
                "Consistency_Index": {"value":float(consistency_index_preds), "description":"Consistency Index: Measures the consistency of slum/non-slum predictions over multiple time steps. A high consistency index indicates reliable predictions over time."},
            }
        }
    }
    
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filename}")

@click.command()
@click.option('--true_t1_shp', type=click.Path(exists=True), help='Path to t1_true shapefile [*.shp]')
@click.option('--pred_t1_shp', type=click.Path(exists=True), help='Path to t1_pred shapefile [*.shp]')
@click.option('--true_t2_shp', type=click.Path(exists=True), help='Path to t2_true shapefile [*.shp]')
@click.option('--pred_t2_shp', type=click.Path(exists=True), help='Path to t2_pred shapefile [*.shp]')
@click.option('--config','-c', type=click.Path(exists=True), default=None, help='Path to configuration file')
@click.option('--output_filename', default=None, help='The json file to save the results to.')
@click.option('--text', default="", help='A textual description of the whole change detection exercise that will be included in the output file.')
@click.help_option('-h', '--help')

def main(true_t1_shp, pred_t1_shp, true_t2_shp, pred_t2_shp, output_filename, text, config):
    if config:
        with open(config, 'r') as ymlfile:
            config_data = yaml.load(ymlfile, Loader=yaml.FullLoader)
            t1_true = config_data['true_t1_shp']
            t1_pred = config_data['pred_t1_shp']
            t2_true = config_data['true_t2_shp']
            t2_pred = config_data['pred_t2_shp']
            text = config_data['text']
            output_filename = config_data['output_filename']
    
    def load_and_preprocess_shapefiles(shapefile_paths, pixels_per_meter=1, output_paths=None):
        # Load the shapefiles and remove large polygons
        gdfs = [remove_large_polygon(gpd.read_file(path)) for path in shapefile_paths]

        # Determine the bounding boxes of all shapefiles
        bboxes = [gdf.total_bounds for gdf in gdfs]

        # Calculate the intersection of the bounding boxes
        min_x = max(bbox[0] for bbox in bboxes)
        min_y = max(bbox[1] for bbox in bboxes)
        max_x = min(bbox[2] for bbox in bboxes)
        max_y = min(bbox[3] for bbox in bboxes)
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

        # Default output paths if not provided
        if output_paths is None:
            output_paths = [f'shapefile{i+1}.png' for i in range(len(shapefile_paths))]

        # Plot and save the shapefiles as grayscale PNGs
        for gdf, output_path in zip(gdfs, output_paths):
            print(f"Pre-processing file {os.path.basename(output_path)}")
            plot_and_save_aligned(gdf, intersection_bounding_box, output_path, width_pixels, height_pixels)

        def load_and_binarize_image(image_path, threshold=128):
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            binary_image = (np.array(image) < threshold).astype(np.uint8) * 255
            binary_image_pil = Image.fromarray(binary_image)
            binary_image_pil.save(image_path)  # Overwrite the grayscale image with the binary image
            return binary_image_pil, binary_image

        binary_images_np = []
        for output_path in output_paths:
            _, binary_image_np = load_and_binarize_image(output_path)
            binary_images_np.append(binary_image_np)

        return binary_images_np

    print("Loading and preprocessing shapefiles ...")
    shapefile_paths = [ t1_true, t1_pred, t2_true, t2_pred ]
    output_dir = os.path.dirname(output_filename)
    output_paths = output_paths = [os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0][:-4] + '.png') 
                                        for path in shapefile_paths]
    binary_images_np = load_and_preprocess_shapefiles(shapefile_paths, pixels_per_meter=1, output_paths=output_paths)
    print("Loading data for T1 ...")
    y_true_t1 = binary_images_np[0]
    y_pred_t1 = binary_images_np[1]
 
    print("Loading data for T2 ...")
    y_true_t2 = binary_images_np[2]
    y_pred_t2 = binary_images_np[3]

    print(20*"#"," Single image metrics ", 20*"#")
    iou_t1, precision_t1, recall_t1, f1_t1 = compute_metrics(y_true_t1, y_pred_t1)
    print(f"Metrics at t1: IoU={iou_t1}, Precision={precision_t1}, Recall={recall_t1}, F1={f1_t1}")
    iou_t2, precision_t2, recall_t2, f1_t2 = compute_metrics(y_true_t2, y_pred_t2)
    print(f"Metrics at t2: IoU={iou_t2}, Precision={precision_t2}, Recall={recall_t2}, F1={f1_t2}")
    print("")
    print(20*"#"," Change metrics ", 20*"#")
    cda, change_iou, tprc = compute_change_metrics(y_true_t1, y_pred_t1, y_true_t2, y_pred_t2)
    print(f"Change Metrics: CDA={cda}, Change IoU={change_iou}, TPRC={tprc}")
    tsac, nsadr, fhcr = compute_area_metrics(y_true_t1, y_pred_t1, y_true_t2, y_pred_t2)
    print(f"Area Metrics: TSAC={tsac}, NSADR={nsadr}, FHCR={fhcr}")

    print("")
    print(20*"#"," Stability indices ", 20*"#")
    true_joined = np.concatenate([y_true_t1.ravel().reshape([-1,1]), 
                                  y_true_t2.ravel().reshape([-1,1])], axis=1)
    temporal_stability_true = compute_temporal_stability(true_joined)
    print(f"Temporal Stability (true): {temporal_stability_true}")
    consistency_index_true = compute_consistency_index(true_joined)
    print(f"Consistency Index (true): {consistency_index_true}")
    predictions_joined = np.concatenate([y_pred_t1.ravel().reshape([-1,1]),
                                         y_pred_t2.ravel().reshape([-1,1])], axis=1)
    temporal_stability_preds = compute_temporal_stability(predictions_joined)
    print(f"Temporal Stability (preds): {temporal_stability_preds}")
    consistency_index_preds = compute_consistency_index(predictions_joined)
    print(f"Consistency Index (preds): {consistency_index_preds}")
    
    save_results(output_filename, text, iou_t1, precision_t1, recall_t1, f1_t1,
                 iou_t2, precision_t2, recall_t2, f1_t2,
                 cda, change_iou, tprc, tsac, nsadr, fhcr,
                 temporal_stability_true, consistency_index_true,
                 temporal_stability_preds, consistency_index_preds)

if __name__ == '__main__':
    main()
