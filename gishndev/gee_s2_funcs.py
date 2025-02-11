# gee_s2_funcs.py
# Funciones para trabajar con imágenes satelitales y la gee
import os
import ee
import geemap
import json
import rasterio
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
from rasterio.features import shapes

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Leer los indices, constantes y bandas de spectral
with open(os.path.join(current_dir, "spectral_indices.json")) as f:
    spectral = json.load(f)
    spectral_indices = spectral["spectral_indices"]

with open(os.path.join(current_dir, "spectral_constants.json")) as f:
    spectral_constants = json.load(f)

with open(os.path.join(current_dir, "spectral_bands.json")) as f:
    spectral_bands = json.load(f)


def mask_s2_clouds(img):
    """
    Masks clouds and cloud shadows in Sentinel-2 images using the SCL band, QA60 band, and cloud probability.
    This function applies a series of masks to a Sentinel-2 image to remove pixels that are likely to be affected by clouds, cloud shadows, cirrus, or snow. It uses the Scene Classification Layer (SCL) band, the QA60 band, and optionally the cloud probability band from the COPERNICUS/S2_CLOUD_PROBABILITY collection.

    :param image: ee.Image
        The Sentinel-2 image to be masked.
    :return: ee.Image
        The masked Sentinel-2 image with values scaled between 0 and 1.
    """
    # Load the cloud probability image collection
    cloud_prob_collection = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(img.geometry())
        .filterDate(img.date(), img.date().advance(1, "day"))
    )

    # Get the first cloud probability image if available
    cloud_prob_img = ee.Algorithms.If(
        cloud_prob_collection.size().gt(0),
        cloud_prob_collection.first(),  # Use cloud probability image if it exists
        None,  # Otherwise, return None to skip cloud probability masking
    )

    # Use the SCL band for scene classification
    scl = img.select("SCL")

    # Create masks based on the SCL band
    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))  # 3 = Cloud shadow
        .And(scl.neq(9))  # 8 = Clouds
        .And(scl.neq(10))  # 9 = Cirrus
    )  # 10 = Snow

    # QA60 mask for clouds and cirrus
    qa = img.select("QA60")
    cloud_bit_mask = ee.Number(1024)  # 2^10 = 1024, 10th bit is clouds
    cirrus_bit_mask = ee.Number(2048)  # 2^11 = 2048, 11th bit is cirrus clouds
    mask_qa60 = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    # Use cloud probability threshold (e.g., clouds if probability > 20%)
    cloud_prob_mask = ee.Image(cloud_prob_img).select("probability").lt(20)

    # Combine the SCL mask, QA60 mask, and cloud probability mask (if available)
    combined_mask = ee.Algorithms.If(
        cloud_prob_img,  # If cloud_prob_image is not None
        scl_mask.And(mask_qa60).And(cloud_prob_mask),  # Combine all masks
        scl_mask.And(
            mask_qa60
        ),  # Use only SCL and QA60 mask if cloud probability image is unavailable
    )

    # Return the masked image, scaled by 10,000 to get values between 0-1
    return img.updateMask(combined_mask)


def create_gdfbounds_from_tif(tif_path):
    """
    Crea un GeoDataFrame a partir de los límites de un archivo TIFF.

    Args:
        tif_path (str): Ruta al archivo TIFF.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con un polígono que representa los límites del archivo TIFF.
    """
    # Leer el archivo TIFF
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs

    # Crear un polígono a partir de los límites
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Crear un GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=crs)

    return gdf


def apply_scale_factorsL8(img):
    """
    Applies scale factors to Landsat 8 imagery bands.
    This function scales the optical and thermal bands of a Landsat 8 image.
    The optical bands are scaled by multiplying by 0.0000275 and adding -0.2.
    The thermal bands are scaled by multiplying by 0.00341802 and adding 149.0.

    :param image: ee.Image
        The input Landsat 8 image to which the scale factors will be applied.
    :return: ee.Image
        The image with scaled optical and thermal bands.
    """
    optical_bands = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = img.select("ST_B.*").multiply(0.00341802).add(149.0)
    return img.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


def index_info(index, properties=["formula"]):
    """
    Retrieve and print information about specified spectral indices.

    :param index: A single index or a list of indices to retrieve information for.
    :type index: str or list of str
    :param properties: A list of properties to retrieve for each index. Default is ["formula"].
                       Possible properties include 'application_domain', 'bands', 'contributor',
                       'date_of_addition', 'formula', 'long_name', 'platforms', 'reference', 'short_name'.
    :type properties: list of str
    """
    if not isinstance(index, list):
        index = [index]

    if not isinstance(properties, list):
        properties = [properties]

    for idx in index:
        properties_dic = {}
        for property in properties:
            properties_dic[property] = spectral_indices[idx][property]
        print(f"'{idx}' info:")
        print(properties_dic)


def compute_index(img, index, params):
    """
    Computes spectral indices for the given image and adds them as bands.

    :param img: The input image to which the spectral indices will be added.
    :type img: ee.Image
    :param index: A list of spectral indices to compute. If a single index is provided, it will be converted to a list.
    :type index: list or str
    :param params: A dictionary of parameters required for computing the indices.
    :type params: dict
    :return: The input image with the computed spectral indices added as bands.
    :rtype: ee.Image
    """
    if not isinstance(index, list):
        index = [index]

    for idx in index:
        formula = spectral_indices[idx]["formula"]
        img = img.addBands(img.expression(formula, params).rename(idx))

    return img


def params_index_s2(index, img):
    """
    Processes spectral indices and returns a dictionary of parameters with their corresponding values or bands.

    :param index: A list of spectral index names or a single spectral index name.
    :type index: list or str
    :param img: An image object from which bands are selected.
    :type img: ee.Image
    :return: A dictionary where keys are parameter names and values are either constants or selected bands from the image.
    :rtype: dict
    """
    # Convertir a lista sino lo es``
    if not isinstance(index, list):
        index = [index]

    # Obtener los parámetros de los índices
    unique_params = []
    for idx in index:
        idx_params = spectral_indices[idx]["bands"]
        total_params = unique_params + idx_params
        unique_params = list(set(total_params))

    ##  Crear las constantes
    param_constants = [
        param for param in unique_params if param in spectral_constants.keys()
    ]
    constants_values = {}
    for constant in param_constants:
        value = spectral_constants[constant]["default"]
        constants_values[constant] = value

    ## Asignar la banda de sentinel a cada params_bands
    param_bands = [
        param for param in unique_params if param not in spectral_constants.keys()
    ]
    s2_param_bands = {}
    for band in param_bands:
        s2_band = spectral_bands[band]["platforms"]["sentinel2a"]["band"]
        s2_param_bands[band] = s2_band

    ## Crear los parámetros
    params = {}
    for param in unique_params:
        if param in constants_values.keys():
            value = constants_values[param]
            params[param] = value
        elif param in s2_param_bands.keys():
            band_name = s2_param_bands[param]
            params[param] = img.select(band_name)
    return params


def extract_values_to_point_per_part(
    samples, stack, num_samples, out_dir, out_file_name
):
    """
    Extracts values to points for each part of the samples and saves the output to a specified directory.

    :param samples: A GeoDataFrame containing the sample points.
    :param stack: The image stack from which to extract values.
    :param num_samples: The number of samples to process in each part.
    :param out_dir: The output directory where the results will be saved.
    """
    # Extract the basename of the input samples
    out_fc_base = f"{out_dir}/{out_file_name}"

    num_parts = (samples.shape[0] // num_samples) + 1
    for i in range(num_parts):
        start_idx = i * num_samples
        end_idx = start_idx + num_samples - 1
        samples_part = samples.iloc[start_idx:end_idx]
        samples_part_ee = geemap.geopandas_to_ee(samples_part)
        geemap.extract_values_to_points(
            samples_part_ee, stack, scale=10, out_fc=f"{out_fc_base}_{i}.csv"
        )


def get_sampled_size(samples, stack, n_samples):
    """
    Get the size of the sampled regions from a stack based on a subset of samples.

    :param samples: A pandas DataFrame containing the sample points.
    :param stack: An Earth Engine Image or ImageCollection to sample from.
    :param n_samples: The number of samples to take from the samples DataFrame.
    :return: The size of the sampled regions as an integer.
    """

    subset = samples.iloc[0:n_samples]
    subset_ee = geemap.geopandas_to_ee(subset)
    sampled = stack.sampleRegions(collection=subset_ee, scale=10, geometries=True)
    return sampled.size().getInfo()


def ee_to_df_sampled(samples, stack, n_samples):
    """
    Subsets the given samples DataFrame, converts it to an Earth Engine FeatureCollection,
    samples the given stack Image using the subset, and returns the sampled data as a DataFrame.

    :param samples: pandas.DataFrame
        A DataFrame containing the sample points with their respective coordinates.
    :param stack: ee.Image
        An Earth Engine Image object from which to sample the data.
    :param n_samples: int
        The number of samples to subset from the samples DataFrame.
    :return: pandas.DataFrame
        A DataFrame containing the sampled data with properties from the Earth Engine Image.
    """

    subset = samples.iloc[0:n_samples]
    subset_ee = geemap.geopandas_to_ee(subset)
    sampled = stack.sampleRegions(collection=subset_ee, scale=10, geometries=True)
    features = sampled.getInfo()["features"]
    data = []
    for feature in features:
        properties = feature["properties"]
        data.append(properties)
    df = pd.DataFrame(data)
    return df


def tif_to_gdf(tif):
    """
    Converts a TIFF raster file to a GeoDataFrame.
    This function reads a TIFF file, processes its raster data, and converts it into a GeoDataFrame.
    The raster data is converted to float32 if it is not in a compatible data type. Only the pixels
    with values greater than 0 are considered for conversion to geometries.

    :param tif: str
        The file path to the TIFF raster file.
    :return: geopandas.GeoDataFrame
        A GeoDataFrame containing the geometries and their associated values extracted from the raster.
    :raises Exception: If there is an error in reading the TIFF file or converting it to a GeoDataFrame.
    """

    # Crear el nombre del archivo tif
    try:
        with rasterio.open(tif) as src:
            image = src.read(1)

            # Convertir el tipo de datos del raster a float32 si no es compatible
            if image.dtype not in ["int16", "int32", "uint8", "uint16", "float32"]:
                image = image.astype("float32")

            mask = image > 0
            results = (
                {"properties": {"value": v}, "geometry": s}
                for s, v in shapes(image, mask=mask, transform=src.transform)
            )
            geoms = list(results)

        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
        return gdf
    except Exception as e:
        print(f"Error al convertir el raster a GeoDataFrame: {e}")
        raise

# other function ..
