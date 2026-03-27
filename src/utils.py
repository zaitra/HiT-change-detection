import matplotlib.pyplot as plt
import torch

SENTINEL2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
FLOGA_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "B8A"]
PRITHVI_INDICES = [1,2,3, 7,11, 12]

def sentinel2_bands_to_indices(bands):
    band_indices = {band: i for i, band in enumerate(SENTINEL2_BANDS)}
    return [band_indices[band] for band in bands if band in band_indices]

