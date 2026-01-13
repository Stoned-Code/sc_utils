from datasets import load_dataset
import lmdb
import pathlib
import shutil
import os
import PIL


def parquet_to_lmdb(dataset, split, output_path,
                    img_key="pixel_values", batch_size=50, ms_scalar=0.1, streaming=True):

    temp_path = pathlib.Path(output_path) / "temp"

    temp_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset, split=None if streaming else split, streaming=streaming)

    #print("Length:", len(ds))

    #length = len(ds)
    if streaming:
        length = ds[split].info.splits[split].num_examples
    else:
        length = len(ds)
    print("Length:", length)

    for i, data in enumerate(ds[split] if streaming else ds):

        print(f"Image {i + 1}/{length}", end="\r")

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
            if t is PIL.PngImagePlugin.PngImageFile:
                img_path = temp_path / f"{i}_temp.png"

            elif t is PIL.JpegImagePlugin.JpegImageFile:
                img_path = temp_path / f"{i}_temp.jpg"

            else:
                img_path = temp_path / f"{i}_temp.jpg"

            img.save(img_path)

    paths = [temp_path / p for p in os.listdir(temp_path)]

    file_sizes = []
    path_length = len(paths)

    for i, p in enumerate(paths):
        print(f"Path {i + 1}/{path_length}", end="\r")

        file_size = os.path.getsize(p)
        file_sizes.append(file_size)

    print()

    map_size = sum(file_sizes)
    map_size = map_size + (map_size * ms_scalar)
    map_size_gb = map_size / (1024 ** 3)
    print(f"Map Size: {round(map_size_gb, 2)}GB")

    #output_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_path), map_size=int(map_size))

    txn = env.begin(write=True)

    path_length = len(paths)

    for i, path in enumerate(paths):
        print(f"Path {i + 1}/{path_length}", end="\r")

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
    p.add_argument("--ms_scalar", help=f"The scalar that decides the percentage of extra storage added to the max storage size. (default:{ms_scalar})")

    args = p.parse_args()

    dataset = args.dataset
    split = args.split
    output_path = args.output
    img_key = args.img_key
    batch_size = args.batch_size
    ms_scalar = args.ms_scalar

    if args.get_splits:
        ds = load_dataset(dataset, streaming=True)
        print("-" * 10)
        print("Splits:")
        [print(f"- \"{s}\"") for s in ds.keys()]
        print("-" * 10)

        exit()

    parquet_to_lmdb(dataset, split, str(output_path), img_key, batch_size, ms_scalar, True)