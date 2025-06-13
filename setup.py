# setup.py
from setuptools import setup

setup(
    name='sc_utils',         # Name of your module
    version='0.2',        # Version number
    py_modules=[
        'create_autoencoder_data', 
    'video_processing', 
    "image_processing", 
    "youtube_download",
    "hf_datasets",
    "create_frame_generator_data",
    "data_processing_webui",
    "file_processing",
    "sc_datasets",
    "sc_numpy",
    "audio_processing"], # Module name (without .py)
)