from datasets import load_dataset
import lmdb
import pathlib
import shutil
import os
import PIL
from PIL import PngImagePlugin, JpegImagePlugin, TiffImagePlugin
from processing.image_processing import set_shortest_length
import tqdm

def parquet_to_lmdb(dataset, split, output_path,
                    img_key="pixel_values", batch_size=50, ms_scalar=0.1,
                    streaming=True, skip_saving=False, min_size=None,
                    delete_temp=True, token=None):

    temp_path = pathlib.Path(output_path) / "temp"

    temp_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset, split=None if streaming else split, streaming=streaming, token=token)

    #print("Length:", len(ds))

    #length = len(ds)
    if streaming:
        length = ds[split].info.splits[split].num_examples
    else:
        length = len(ds)
    print("Length:", length)
    if not skip_saving:
        for i, data in tqdm.tqdm(enumerate(ds[split] if streaming else ds), total=length, desc="Saving Image"):

            # print(f"Saving Image {i + 1}/{length}", end="\r")

            try:
                img = data[img_key]

            except KeyError as e:
                print("Available Keys:")
                print(data.keys())
                img_key = input("Type out key: ").strip()
                img = data[img_key]
                exit()
            
            finally:
                t = type(img)
                
                if t is PngImagePlugin.PngImageFile:
                    img_path = temp_path / f"{i}_temp.png"

                elif t is JpegImagePlugin.JpegImageFile:
                    img_path = temp_path / f"{i}_temp.jpg"
                
                elif t is TiffImagePlugin.TiffImageFile:
                    img_path = temp_path / f"{i}_temp.tif"

                else:
                    img_path = temp_path / f"{i}_temp.jpg"

                minimum = min(img.size)

                if min_size is not None and minimum > min_size:
                    img = set_shortest_length(img, min_size)
                
                if not os.path.exists(img_path):
                    img.save(img_path)

    paths = [temp_path / p for p in os.listdir(temp_path)]

    file_sizes = []
    path_length = len(paths)

    for i, p in tqdm.tqdm(enumerate(paths), total=path_length, desc="Getting Path Size"):
        # print(f"Getting Path Size {i + 1}/{path_length}", end="\r")

        file_size = os.path.getsize(p)
        file_sizes.append(file_size)

    print()

    map_size = sum(file_sizes)
    print("Sum:", map_size)
    print("MS Scalar:", ms_scalar)

    map_size = map_size + (map_size * ms_scalar)
    map_size_gb = map_size / (1024 ** 3)
    print(f"Map Size: {round(map_size_gb, 2)}GB")

    #output_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_path), map_size=int(map_size))

    txn = env.begin(write=True)

    path_length = len(paths)

    for i, path in tqdm.tqdm(enumerate(paths), total=len(path_length), desc="Adding Path"):
        # print(f"Adding Path {i + 1}/{path_length}", end="\r")

        with open(path, "rb") as f:
            img_data = f.read()

        txn.put(f"{i}".encode(), img_data)
        # img.close()

        del img_data

        if (i + 1) % batch_size == 0:
            txn.commit()
            txn = env.begin(write=True)
    try:
        txn.commit()
    except Exception as ex:
        print(ex)
    finally:
        print()

    # with env.begin() as txn:
    #     value = txn.get(b'0')
    #     print(value)

    env.close()
    if delete_temp:
        print("Deleting temp paths...")
        shutil.rmtree(temp_path)
    print(f"Finished creating dataset {output_path}")


if __name__ == "__main__":
    import argparse

    dataset = "/mnt/data/Vision/NSFW/NSFW"
    split = "train"
    output_path = pathlib.Path(dataset) / f"unstructured_{split}"
    img_key = "pixel_values"
    batch_size = 100
    ms_scalar = 0.1
    min_size = None
    token = None

    p = argparse.ArgumentParser()

    p.add_argument("--split", help=f"The data split (default: {split})",
                   type=str, default=split)
    p.add_argument("--dataset", help=f"The dataset path or ID to an image parquet dataset. (default: {dataset}",
                   type=str, default=dataset)
    p.add_argument("--output", "-O", help=f"The output directory for the LMDB dataset. (default: {str(output_path)})",
                   type=str, default=str(output_path))
    p.add_argument("--img_key", help=f"The key for the image column. (default: {img_key})",
                   type=str, default=img_key)
    p.add_argument("--batch_size", help=f"The amount of images to batch up before commiting them to a file. (default: {batch_size}",
                   type=int, default=batch_size)
    p.add_argument("--get_splits", help="Use in order to retrieve a list of splits from the dataset.",
                   action="store_true")
    p.add_argument("--ms_scalar", help=f"The scalar that decides the percentage of extra storage added to the max storage size. (default:{ms_scalar})",
                   type=float,
                   default=ms_scalar)
    p.add_argument("--skip_saving", help="Whether to skip the saving of parquet images to raw images on the system.",
                   action="store_true")
    
    p.add_argument("--min_size", help=f"The minimum size to set the shortest length of an image to. (default: {str(min_size)})",
                   type=int, default=min_size)

    p.add_argument("--no_delete", help=f"If enabled, files won't be deleted",
                   action="store_true")
    
    p.add_argument("--token", help=f"The token for for huggingface. (default: {token})", type=str, default = token)


    
    
    args = p.parse_args()

    dataset = args.dataset
    split = args.split
    output_path = args.output
    img_key = args.img_key
    batch_size = args.batch_size
    ms_scalar = args.ms_scalar
    min_size = args.min_size
    token = args.token

    if token is not None and token.endswith(".key"):
        with open(token, "r") as f:
            token = f.read().strip()

    if args.get_splits:
        ds = load_dataset(dataset, streaming=True)
        print("-" * 10)
        print("Splits:")
        [print(f"- \"{s}\"") for s in ds.keys()]
        print("-" * 10)

        exit()

    parquet_to_lmdb(dataset, split, str(output_path), img_key, batch_size, ms_scalar, True, args.skip_saving, min_size, not args.no_delete, token)