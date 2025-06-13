import os
from datasets import load_dataset
from PIL import Image

def download_images_from_hf_dataset(dataset_id: str, pull_token: str = None, download_folder: str="data/images", image_column: str="image", maximum=-1):
    """
    Downloads images from a Hugging Face image dataset to a specified folder.

    Args:
        dataset_id (str): The ID of the Hugging Face dataset, e.g., "zaibutcooler/artistic".
        pull_token (str, optional): Authentication token for private datasets. Defaults to None.
        download_folder (str): The path to the folder where images will be saved.
        image_column (str): The name of the column in the dataset that contains the images.

    Raises:
        ValueError: If the specified image column does not contain PIL Image objects.
        KeyError: If the image_column does not exist in the dataset.
        Exception: If there are issues loading the dataset (e.g., invalid dataset ID or insufficient permissions).

    Note:
        Requires the `datasets` and `Pillow` libraries. Install them using:
            pip install datasets Pillow
        For private datasets, provide a valid pull token obtained from Hugging Face.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_id, token=pull_token)


    # Helper function to save images from a dataset to a folder
    def save_images(dataset, folder):
        format_to_ext = {
            'JPEG': 'jpg',
            'PNG': 'png',
            'BMP': 'bmp',
            'GIF': 'gif',
        }
        overall_amt = len(dataset)

        for index, sample in enumerate(dataset):
            print(f"Downloading {index + 1}/{overall_amt}...", end="\r")
            image = sample[image_column]
            if not isinstance(image, Image.Image):
                raise ValueError(f"The column '{image_column}' does not contain PIL Image objects.")
            format = image.format
            ext = format_to_ext.get(format, 'jpg')  # Default to 'jpg' if format is unknown
            filename = f"image_{index:06d}.{ext}"
            image.save(os.path.join(folder, filename))
            if maximum > 0 and index == maximum - 1:
                break
        print(f"Downloading {index + 1}/{overall_amt}...", end="\r")          
    # Handle DatasetDict (multiple splits) or Dataset (single split)
    if isinstance(dataset, dict):  # DatasetDict
        for split_name, split_dataset in dataset.items():
            split_folder = os.path.join(download_folder, split_name)
            os.makedirs(split_folder, exist_ok=True)
            save_images(split_dataset, split_folder)
    else:  # Dataset
        os.makedirs(download_folder, exist_ok=True)
        save_images(dataset, download_folder)

if __name__ == "__main__":
    import sys

    download_images = "--image" in sys.argv

    if download_images:
        img_column = "image" if "--img_col" not in sys.argv else sys.argv[sys.argv.index("--img_col") + 1]
        download_folder = "data/images" if "--output_folder" not in sys.argv else sys.argv[sys.argv.index("--output_folder") + 1]
        maximum_images = -1 if "--maximum" not in sys.argv else int(sys.argv[sys.argv.index("--maximum") + 1])
        token = None if "--token" not in sys.argv else sys.argv[sys.argv.index("--token") + 1]
        dataset_id = sys.argv[-1]

        download_images_from_hf_dataset(dataset_id, token, download_folder, img_column, maximum_images)