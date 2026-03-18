import geopandas as gpd
import click

def calculate_area_difference(original_gdf, simplified_gdf):
    original_area = original_gdf['geometry'].area.sum()
    simplified_area = simplified_gdf['geometry'].area.sum()
    return abs(original_area - simplified_area) / original_area

@click.command(context_settings=dict(help_option_names=['-h', '--help']), help="""
This script loads a number of shapefiles and evaluates the effect of a set of simplification factors on them.
The user provides the shapefiles to investigate and the simplification factors to evaluate.
Example usage:
    python simplify_shapefiles.py --shapefiles /path/to/shapefile1 -- shapefiles /path/to/shapefile2 --tolerances 10 --tolerances 1 --tolerances 0.1
""")
@click.option('--shapefiles', required=True, multiple=True, help='Paths to the shapefiles')
@click.option('--tolerances', required=True, type=float, multiple=True, help='Tolerances for simplification')
@click.help_option('-h', '--help')

def main(shapefiles, tolerances):
    for shapefile in shapefiles:
        gdf = gpd.read_file(shapefile)
        print(f"===== {shapefile} =====")
        for tolerance in tolerances:
            simplified_gdf = gdf.copy()
            simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(tolerance, preserve_topology=True)
            area_difference = calculate_area_difference(gdf, simplified_gdf)
            print(f'Area difference with tolerance {tolerance}: {area_difference:.5%}')


if __name__ == "__main__":
    main()