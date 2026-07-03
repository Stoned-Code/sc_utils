import pathlib
import lmdb
import glob
import os
import pickle
from PIL import Image
import numpy as np
# import gc
import time

import pandas as pd
from sc_utils.processing.image_processing import get_hash_from_path
import tqdm

def fix_img_lmdb(path, output, batch_size):
    data = os.listdir(path)
    sizes = []

    for i, p in enumerate(data):
        file_size = os.path.getsize(p)
        sizes.append(file_size)

    map_size = sum(sizes)
    map_size_gb = map_size / (1024 ** 3)

    print(f"Map Size: {round(map_size_gb, 2)}GB")

    old_env = lmdb.open(path)
    old_txn = old_env.begin()

    

def to_lmdb(paths, output, batch_size, mp_scalar = 0.1):
    file_sizes = []
    path_length = len(paths)
    print("Creating lmdb dataset...")
    for i, p in enumerate(paths):
        print(f"Getting Size {i + 1}/{path_length}", end="\r")
        file_size = os.path.getsize(p)
        file_sizes.append(file_size)

    print()

    map_size = sum(file_sizes)
    map_size = map_size + (map_size * mp_scalar)
    map_size_gb = map_size/(1024 **3)
    print(f"Map Size: {round(map_size_gb, 2)}GB")

    output_path = pathlib.Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_path), map_size=int(map_size))

    txn = env.begin(write=True)
    #with env.begin(write=True) as txn:
    path_length = len(paths)
    for i, path in enumerate(paths):
        print(f"Adding File {i + 1}/{path_length}", end="\r")
        #img = Image.open(path)
        #img_array = np.array(img)
        #img_data = pickle.dumps(img)
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


if __name__ == "__main__":
    import argparse
    import random

    p = argparse.ArgumentParser()

    p.add_argument("paths", nargs=argparse.REMAINDER)
    p.add_argument("--output", "-O", type=str, default="data/lmdb")
    p.add_argument("--test_split", type=float, default = None)
    p.add_argument("--batch_size", "-b", type=int, default=100)
    p.add_argument("--ms_scalar", type=float, default = 0.1)
    p.add_argument("--shuffle", action="store_true")

    args = p.parse_args()

    paths = []

    for p in args.paths:
        if "*" in p:
            paths.extend(glob.glob(p))
        else:
            paths.append(p)

    
    df = pd.DataFrame({"path": paths})
    df["hash"] = 0
    errored = 0
    length = len(df)
    for i, row in df.iterrows():
        print(f"Getting Hash {i + 1}/{length}", end="\r")
        try:
            h, _, _, _ = get_hash_from_path(row["path"])
        except PermissionError as ex:
            errored = 0
            continue
        df.at[i, "hash"] = h
    df = df[df["hash"] != 0]
    print()
    if errored > 0:
        print(f"{errored} errored out...")
    df = df.drop_duplicates(subset=["hash"], ignore_index=True)

    paths = df["path"].values

    del df

    if args.shuffle:
        random.shuffle(paths)

    if args.test_split is None:
        print("Creating dataset.")
        to_lmdb(paths, args.output, args.batch_size, args.ms_scalar)
    
    else:
        train_amt = int((1.0 - args.test_split) * len(paths))
        train_paths = paths[:train_amt]
        test_paths = paths[train_amt:]#[p for p in paths if p not in train_paths]

        out_path = pathlib.Path(args.output)
        test_path = out_path / "test"
        train_path = out_path / "train"

        print("Creating Test split...")
        to_lmdb(test_paths, str(test_path), args.batch_size, args.ms_scalar)
        print("Creating Train split...")
        to_lmdb(train_paths, str(train_path), args.batch_size, args.ms_scalar)