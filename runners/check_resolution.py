## This code checks the spatial resolution for inputs in a given directoy
# Standard use: python -m slumworldML.runners.check_resolution -d /home/yuting/data/raw/Bobo/Gamma_0.2/ -f input_x
# must have .png file and .png.aux.xml file

from PIL import Image
import os
import argparse
import xml.etree.ElementTree as ET
Image.MAX_IMAGE_PIXELS = None


def get_png_info(png_file_path):
    """
    Checks the pixel resolution (width and height), DPI of a PNG file,
    and attempts to read spatial resolution (cell size) from an associated .aux.xml file
    located in a parallel 'auxiliary_dir'.

    Args:
        png_file_path (str): The full path to the PNG file (e.g., /parent_dir/image_dir/filename.png).

    Returns:
        dict: A dictionary containing image information (pixel_resolution, dpi,
              spatial_resolution) or None if the PNG file is not found or invalid.
              spatial_resolution will be a tuple (x_cell_size, y_cell_size) or None.
    """
    if not os.path.exists(png_file_path):
        print(f"Error: PNG file not found at {png_file_path}")
        return None

    image_info = {
        'pixel_resolution': None,
        'dpi': None,
        'spatial_resolution': None # This will store cell size from XML
    }

    try:
        # --- Read PNG information using Pillow ---
        with Image.open(png_file_path) as img:
            if img.format != 'PNG':
                print(f"Warning: File is not in PNG format ({img.format})")
                # Decide if you want to proceed with non-PNG files or return None
                # For now, we'll return None if not strictly PNG as per previous logic
                return None

            # Get pixel dimensions
            image_info['pixel_resolution'] = img.size

            # Get DPI information
            image_info['dpi'] = img.info.get('dpi')

        # --- Attempt to read information from associated .aux.xml file ---
        # Derive the path to the associated .aux.xml file based on the expected structure
        # Assuming the structure is /parent_dir/image_dir/filename.png
        # and the aux file is at /parent_dir/auxiliary_dir/filename.png.aux.xml

        # Get the directory containing the image_dir (the parent directory like version007)
        parent_dir = os.path.dirname(os.path.dirname(png_file_path))
        # Get the base filename with .png extension
        png_basename = os.path.basename(png_file_path)
        # Construct the path to the aux xml file in the auxiliary_dir
        aux_xml_path = os.path.join(parent_dir, 'auxiliary_dir', png_basename + '.aux.xml')


        if os.path.exists(aux_xml_path):
            print(f"Found associated AUX XML file: {aux_xml_path}")
            try:
                tree = ET.parse(aux_xml_path)
                root = tree.getroot()

                # Look for GeoTransform element which often contains cell size
                # The structure can vary, this is based on common GDAL AUX format
                geotransform_element = root.find('GeoTransform')
                if geotransform_element is not None and geotransform_element.text:
                    try:
                        # GeoTransform string format:
                        # top_left_x, pixel_width, rotation_x, top_left_y, rotation_y, pixel_height
                        transform_values = [float(val) for val in geotransform_element.text.split(',')]
                        if len(transform_values) == 6:
                            # pixel_width is the x cell size
                            x_cell_size = transform_values[1]
                            # pixel_height is the y cell size (often negative)
                            y_cell_size = abs(transform_values[5]) # Use absolute value for size
                            image_info['spatial_resolution'] = (x_cell_size, y_cell_size)
                            print(f"Extracted Spatial Resolution (Cell Size) from XML: {x_cell_size}, {y_cell_size}")
                        else:
                            print(f"Warning: GeoTransform element has unexpected number of values: {len(transform_values)}")
                    except ValueError:
                        print(f"Warning: Could not parse GeoTransform values as floats: {geotransform_element.text}")
                else:
                     print("GeoTransform element not found or is empty in AUX XML.")

                # You could add more XML parsing logic here to look for other tags
                # depending on the expected structure of your .aux.xml files.
                # For example, looking for Metadata/MDI tags with specific keys.
                # Example:
                # metadata = root.find('Metadata')
                # if metadata is not None:
                #     for mdi in metadata.findall('MDI'):
                #         key = mdi.get('key')
                #         value = mdi.text
                #         print(f"Metadata Item: {key}={value}")


            except ET.ParseError as e:
                print(f"Error parsing AUX XML file {aux_xml_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing AUX XML: {e}")
        else:
            print(f"No associated AUX XML file found at {aux_xml_path}")


        return image_info

    except FileNotFoundError:
        # This case is already handled by os.path.exists, but good for robustness
        print(f"Error: PNG file not found at {png_file_path}")
        return None
    except Exception as e:
        # Catch other potential errors (e.g., corrupted PNG file, not an image)
        print(f"Error processing PNG file {png_file_path}: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Check the resolution (pixels and DPI) of a PNG file and spatial resolution from associated AUX XML based on directory structure.')

    # Add argument for the parent directory path
    parser.add_argument(
        '-d', '--directory',
        type=str,
        required=True, # Make the directory argument required
        help='Path to the parent directory (e.g., version007).'
    )

    # Add argument for the base filename (without extension)
    parser.add_argument(
        '-f', '--filename',
        type=str,
        required=True, # Make the filename argument required
        help='Base filename of the image (e.g., input_x, input_y, input_z).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the directory path and filename from the parsed arguments
    parent_directory = args.directory
    base_filename = args.filename

    # Construct the full path to the PNG file based on the expected structure
    png_file_path = os.path.join(parent_directory, 'image_dir', base_filename + '.png')

    # Call the function with the constructed PNG file path
    info = get_png_info(png_file_path)

    # Print the result
    if info:
        print("\n--- Image Information ---")
        if info['pixel_resolution']:
            print(f"Pixel Resolution: {info['pixel_resolution'][0]}x{info['pixel_resolution'][1]}")
        if info['dpi']:
            print(f"DPI (Dots Per Inch): {info['dpi'][0]}x{info['dpi'][1]}")
        else:
            print("DPI information not found in PNG metadata.")

        if info['spatial_resolution']:
             print(f"Spatial Resolution (Cell Size): X={info['spatial_resolution'][0]}, Y={info['spatial_resolution'][1]}")
             print("Note: Units depend on the spatial reference system defined in the XML/associated files.")
        else:
            print("Spatial Resolution (Cell Size) not found in associated AUX XML.")

    else:
        print(f"Could not retrieve information for {png_file_path}.")
