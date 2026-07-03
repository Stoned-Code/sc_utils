# setup.py
from setuptools import setup

setup(
    name='sc_utils',         # Name of your module
    version='0.2',        # Version number
    py_modules=[
    'sc_utils.create_autoencoder_data',
    "sc_utils.processing",
    # 'video_processing', 
    # "image_processing", 
    # "file_processing",
    # "audio_processing",
    #"data_processing_webui",
    "sc_utils.parquet_to_lmdb",
    "sc_utils.youtube_download",
    "sc_utils.hf_datasets",
    "sc_utils.create_frame_generator_data",
    "sc_utils.create_key",
    "sc_utils.sc_datasets",
    "sc_utils.sc_numpy",
    "sc_utils.sc_nn",
    "sc_utils.create_lmdb",
    "sc_utils.cuda_check"], # Module name (without .py)
)