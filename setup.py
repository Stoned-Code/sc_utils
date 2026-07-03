# setup.py
from setuptools import setup

setup(
    name='sc_utils',         # Name of your module
    version='0.2',        # Version number
    py_modules=[
    'create_autoencoder_data',
    "processing",
    # 'video_processing', 
    # "image_processing", 
    # "file_processing",
    # "audio_processing",
    #"data_processing_webui",
    "parquet_to_lmdb",
    "youtube_download",
    "hf_datasets",
    "create_frame_generator_data",
    "create_key",
    "sc_datasets",
    "sc_numpy",
    "sc_nn",
    "create_lmdb",
    "cuda_check"], # Module name (without .py)
)