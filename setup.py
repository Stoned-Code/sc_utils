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
    "youtube_download",
    "hf_datasets",
    "create_frame_generator_data",
    "data_processing_webui",
    "sc_datasets",
    "sc_numpy",
    "create_lmdb"], # Module name (without .py)
)