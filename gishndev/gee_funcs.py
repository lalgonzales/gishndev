"""This module contains functions for working with satellite images and Google Earth Engine."""

import os
import ee
import json

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Leer los indices, constantes y bandas de spectral
with open(os.path.join(current_dir, "spectral_indices.json"), encoding="utf-8") as f:
    spectral = json.load(f)
    spectral_indices = spectral["spectral_indices"]

with open(os.path.join(current_dir, "spectral_constants.json"), encoding="utf-8") as f:
    spectral_constants = json.load(f)

with open(os.path.join(current_dir, "spectral_bands.json"), encoding="utf-8") as f:
    spectral_bands = json.load(f)


def mask_s2_clouds(img):
    """
    Masks clouds and cloud shadows in Sentinel-2 images using the SCL band, QA60 band, and cloud probability.

    This function applies a series of masks to a Sentinel-2 image to remove pixels that are likely to be affected by clouds, cloud shadows, cirrus, or snow. It uses the Scene Classification Layer (SCL) band, the QA60 band, and optionally the cloud probability band from the COPERNICUS/S2_CLOUD_PROBABILITY collection.

    Args:
        img (ee.Image): The Sentinel-2 image to be masked.

    Returns:
        ee.Image: The masked Sentinel-2 image with values scaled between 0 and 1.
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


def apply_scale_factorsL8(img):
    """
    Applies scale factors to Landsat 8 imagery bands.

    This function scales the optical and thermal bands of a Landsat 8 image.
    The optical bands are scaled by multiplying by 0.0000275 and adding -0.2.
    The thermal bands are scaled by multiplying by 0.00341802 and adding 149.0.

    Args:
        img (ee.Image): The input Landsat 8 image to which the scale factors will be applied.

    Returns:
        ee.Image: The image with scaled optical and thermal bands.
    """
    optical_bands = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = img.select("ST_B.*").multiply(0.00341802).add(149.0)
    return img.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


def index_info(index, properties=None):
    """
    Retrieve and print information about specified spectral indices.

    Args:
        index (str or list of str): A single index or a list of indices to retrieve information for.
        properties (list of str, optional): A list of properties to retrieve for each index. Default is ["formula"].
                                            Possible properties include 'application_domain', 'bands', 'contributor',
                                            'date_of_addition', 'formula', 'long_name', 'platforms',
                                            'reference', 'short_name'.
    """
    if properties is None:
        properties = ["formula"]

    if not isinstance(index, list):
        index = [index]

    if not isinstance(properties, list):
        properties = [properties]

    for idx in index:
        properties_dic = {}
        for prop in properties:
            properties_dic[prop] = spectral_indices[idx][prop]
        print(f"'{idx}' info:")
        print(properties_dic)


def compute_index(img, index, params):
    """
    Computes spectral indices for the given image and adds them as bands.

    Args:
        img (ee.Image): The input image to which the spectral indices will be added.
        index (list or str): A list of spectral indices to compute. If a single index is provided, it will be converted to a list.
        params (dict): A dictionary of parameters required for computing the indices.

    Returns:
        ee.Image: The input image with the computed spectral indices added as bands.
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

    Args:
        index (list or str): A list of spectral index names or a single spectral index name.
        img (ee.Image): An image object from which bands are selected.

    Returns:
        dict: A dictionary where keys are parameter names and values are either constants or selected bands from the image.
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
