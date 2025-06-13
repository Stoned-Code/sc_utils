import pandas as pd
import glob
import os
import gc
from image_processing import multi_square_crop, process_image
import pathlib
from sc_datasets import shuffle_dataset
from PIL import Image
import numpy as np
from enum import Enum
import random
from datasets import Dataset


class Split(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def load_metadata(path, callback=None):
    #print("Test Path:", path)
    _df = pd.read_csv(path)
    _df["path"] = _df["file"].apply(lambda p: os.path.join(*os.path.split(path)[:-1], "frames", p))
    _df["parent"] = os.path.split(path)[-2]
    
    _df["y_path"] = "N/A"
    _df["y_hash"] = "N/A"
    _df["y_blackness"] = -1.0
    _df["y_whiteness"] = -1.0

    overall_amt = len(_df)
    print("Loading:", path)
    for i, row in _df.iterrows():
        # if i == overall_amt - 1:
        #     continue

        _df.at[i, "y_path"] = _df.at[(i + 1) % overall_amt, "path"]
        _df.at[i, "y_hash"] = _df.at[(i + 1) % overall_amt, "hash"]
        _df.at[i, "y_blackness"] = _df.at[(i + 1) % overall_amt, "blackness"]
        _df.at[i, "y_whiteness"] = _df.at[(i + 1) % overall_amt, "whiteness"]

        print(f"{i + 1}/{overall_amt}", end="\r")
        if callback != None:
            callback(i + 1, overall_amt)
    return _df

def load_metadata_paths(paths, callback=None):
    dfs = []
    
    current_frame = 0
    overall_frames = 0
    overall_amt = len(paths)
    fp = ""

    def md_cb(cur_frame, overall_frame):
        nonlocal current_frame, overall_amt, fp
        # current_frame = cur_frame
        # overall_frames = overall_frame
        if callback != None:
            cb_stats = f"Loading Metadata: {current_frame}/{overall_amt} | {cur_frame}/{overall_frame} {fp}"
            callback(cb_stats)

    for idx, p in enumerate(paths):
        df = load_metadata(p, md_cb)
        dfs.append(df)
        current_frame = idx + 1
        
        if callback != None:
            fp = os.path.split(df.at[0, "parent"])[-1]

    
    dfs = pd.concat(dfs)
    dfs.reset_index(inplace=True, drop=True)
    del df
    gc.collect()
    return dfs

def get_segment_count(amt, overall_amt):
    segs = 1

    while overall_amt / segs > amt:
        segs += 1
    
    return segs


def get_segment_length(amt, overall_amt):
    seg_len = overall_amt / amt
    if seg_len % 1 != 0:
        seg_len = int(seg_len) + 1
    else:
        seg_len = int(seg_len)
    
    return seg_len


def save_dataset(_df, scale, output_dir, prefix="frame_gen", suffix="train", segment_amt = -1, segment_len = -1, shuffle=True, 
                pad_to_square=True, crop_amt = -1, crop_thresh = 1, crop_offset = 1, lower_crop_segments = 2, normalize=False, 
                invert=False, use_grayscale=False, callback_save=None, callback_row_complete=None):
    if not output_dir.endswith("/") and not output_dir.endswith("\\"):
        output_dir += "/"
    
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = [], []
    if segment_len != -1:
        segment_amt = get_segment_count(segment_len, len(_df))
    elif segment_amt != -1:
        segment_len = get_segment_length(segment_amt, len(_df))

    segment = 1
    def save_data():
        nonlocal X, y, segment
        X = np.array(X)
        y = np.array(y)

        if shuffle:
            X, y = shuffle_dataset(X, y)
        X_path = output_dir / f"{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy"
        y_path = output_dir / f"{prefix}_{scale[0]}x{scale[1]}_{suffix}_y_{segment}.npy"
        
        print(f"Saved X Segment: {X_path},", f"Saved y Segment: {y_path}")
        print("X Shape:", X.shape)
        print("Y Shape:", y.shape)
        np.save(X_path, X)
        np.save(y_path, y)
        
        if callback_save != None:
            callback_save(X_path, y_path, segment)

        segment += 1

        del X, y, X_path, y_path
        gc.collect()

        X, y = [], []

    def add_data(inputs, outputs):
        nonlocal X, y
        X.append(inputs)
        y.append(outputs)

        if segment_amt != -1 and len(X) == segment_len:
            save_data()

    overall_amt = len(_df)
    
    if segment_len != -1:
        print(f"Segment Length: {segment_len}")
    # print(_df.columns)
    for i, row in _df.iterrows():
        X_img = Image.open(row["path"]).copy()
        y_img = Image.open(row["y_path"]).copy()
        
        ratio = max([row["width"], row["height"]]) / min([row["width"], row["height"]])
        if crop_amt > 0:
            # X.append(X_img)
            # y.append(y_img)

            crop_squares_X = multi_square_crop(X_img, crop_amt)
            crop_squares_y = multi_square_crop(y_img, crop_amt)
            #print("Crop X:", type(crop_squares_X), crop_amt)
            for crop_X, crop_y in zip(crop_squares_X, crop_squares_y):
                _X, _, _, _, _ = process_image(crop_X, scale, use_grayscale, invert, pad_to_square, normalize)
                _y, _, _, _, _ = process_image(crop_y, scale, use_grayscale, invert, pad_to_square, normalize)
                #_X = _X.copy().resize(scale)
                #_y = _y.copy().resize(scale)
                add_data(_X, _y)
            #X_img, _, _, _, _ = process_image(X_img, scale, use_grayscale, invert, pad_to_square, normalize)
            #y_img, _, _, _, _ = process_image(y_img, scale, use_grayscale, invert, pad_to_square, normalize)
            # X_img = X_img.resize(scale)
            # y_img = y_img.resize(scale)
            #add_data(X_img, y_img)

        else:
            if crop_thresh != -1:
                crop_segs = int(ratio) + crop_offset
            
            if ratio > 1 and ratio < crop_thresh:
                crop_segs = lower_crop_segments
            
            crop_squares_X = list(multi_square_crop(X_img, crop_segs))
            crop_squares_y = list(multi_square_crop(y_img, crop_segs))

            for crop_X, crop_y in zip(crop_squares_X, crop_squares_y):
                crop_X, _, _, _, _ = process_image(crop_X, scale, use_grayscale, invert, pad_to_square, normalize)
                crop_y, _, _, _, _ = process_image(crop_y, scale, use_grayscale, invert, pad_to_square, normalize)
                # crop_X = crop_X.resize(scale)
                # crop_y = crop_y.resize(scale)
                add_data(crop_X, crop_y)
        print(f"{i + 1}/{overall_amt}", end="\r")

        if callback_row_complete != None:
            callback_row_complete(i + 1, overall_amt, segment)
    if len(X) > 0 and segment_amt != -1:
        save_data()
    elif segment_amt == -1 and segment_len == -1:
        X = np.array(X)
        y = np.array(y)

        if shuffle:
            X, y = shuffle_dataset(X, y)
        X_path = output_dir / f"{prefix}_{scale[0]}x{scale[1]}_{suffix}_X.npy"
        y_path = output_dir / f"{prefix}_{scale[0]}x{scale[1]}_{suffix}_y.npy"

        np.save(X_path, X)
        np.save(y_path, y)
        print(f"Saved X Segment: {X_path},", f"Saved y Segment: {y_path}")
    print(f"{i + 1}/{overall_amt}")


def push_to_huggingface(df, huggingface_id, push_token, split: Split, shuffle=True):
    """
    Uploads a dataset of image pairs and additional columns to Hugging Face.

    Parameters:
    - df: pandas DataFrame with 'path', 'y_path', and other columns
    - scale: Tuple (width, height) to resize images
    - huggingface_id: String, Hugging Face repository ID (e.g., 'username/dataset_name')
    - push_token: String, Hugging Face authentication token
    - split: Split enum, specifies the dataset split ('train', 'validation', or 'test')
    - shuffle: Boolean, whether to shuffle the data before uploading (default: True)
    """
    # Get all columns except 'path' and 'y_path'
    additional_columns = [col for col in df.columns if col not in ["path", "y_path"]]

    data_list = []

    # Process each row
    for i, row in df.iterrows():
        # Load and resize images
        input_img = Image.open(row["path"])
        target_img = Image.open(row["y_path"])
        
        # Create dictionary with images and all additional columns
        data_dict = {
            "input_image": input_img,
            "target_image": target_img,
            **{col: row[col] for col in additional_columns}
        }
        data_list.append(data_dict)
        input_img.close()
        target_img.close()
        print(f"Processed {i + 1}/{len(df)}", end="\r")

    # Shuffle if requested
    if shuffle:
        random.shuffle(data_list)

    # Create and upload dataset
    dataset = Dataset.from_list(data_list)
    dataset.push_to_hub(huggingface_id, split=split.value, token=push_token)

    print(f"\nUploaded {len(df)} images with additional columns to {huggingface_id} in {split.value} split.")


if __name__ == "__main__":
    import sys
    import argparse

    p = argparse.ArgumentParser("Frame Geration Dataset Creator")
    p.add_argument("--blackness_thresh", type=float, default=-1.0)
    p.add_argument("--whiteness_thresh", type=float, default=-1.0)
    p.add_argument("--segment_len", type=int, default=-1)
    p.add_argument("--prefix", type=str, default="frame_gen")
    p.add_argument("--crop_amt", type=int, default = -1)
    p.add_argument("--crop_offset", type=int, default=1)
    p.add_argument("--crop_thresh", type = float, default=1)
    p.add_argument("--scale", type=int, default = 240)
    p.add_argument("--output_dir", type=str, default="./data")
    p.add_argument("--test_ratio", type=float, default = 0.2)
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--no_padding", action="store_true")
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--normalize", action = "store_true")
    p.add_argument("--push_token", type=str, default=None)
    p.add_argument("--hf_id", type=str, default=None)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args()
    
    # path = sys.argv[-1]
    print(args.rest)
    paths = args.rest
    
    df = load_metadata_paths(paths)

    df = df[df["y_path"] != "N/A"] # Filter out NA paths
    df = df[df["hash"] != df["y_hash"]]
    print("Dropping duplicates of the same hash.")
    df = df.drop_duplicates(subset=["hash", "y_hash"])

    if args.blackness_thresh != -1:
        print(f"Filtering out blacks above thresh: {args.blackness_thresh}")
        df = df[(df["blackness"] <= args.blackness_thresh) & (df["y_blackness"] <= args.blackness_thresh)]
    
    if args.whiteness_thresh != -1:
        print(f"Filtering out whites above thresh: {args.whiteness_thresh}")
        df = df[(df["whiteness"] <= args.whiteness_thresh) & (df["y_whiteness"] <= args.whiteness_thresh)]

    df = df.sample(frac=1)
    print("Creating training split...")
    train_df = df.sample(frac=1 - args.test_ratio)
    print("Creating test split...")
    test_df = df.drop(train_df.index)
    print("Creating validation split...")
    val_df = test_df.sample(frac=0.5)
    test_df = test_df.drop(val_df.index)
    print("Finished splitting data frames!")
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    if args.push_token == None:
        print("No push token, creating numpy array.")
        segment_amt = get_segment_count(args.segment_len, len(train_df)) if args.segment_len > 0 else args.segment_len
        save_dataset(test_df, (args.scale, args.scale), args.output_dir, args.prefix, "test", segment_amt, -1, not args.no_shuffle, not args.no_padding, args.crop_amt, 
                    args.crop_thresh, args.crop_offset, 2, args.normalize, args.invert, args.grayscale)
        
        save_dataset(val_df, (args.scale, args.scale), args.output_dir, args.prefix, "val", segment_amt, -1, not args.no_shuffle, not args.no_padding, args.crop_amt,
                    args.crop_thresh, args.crop_offset, 2, args.normalize, args.invert, args.grayscale)

        save_dataset(train_df, (args.scale, args.scale), args.output_dir, args.prefix, "train", segment_amt, -1, not args.no_shuffle, not args.no_padding, args.crop_amt,
                    args.crop_thresh, args.crop_offset, 2, args.normalize, args.invert, args.grayscale)
    else:
        assert args.push_token != None and args.hf_id != None

        push_token = args.push_token
        hf_id = args.hf_id

        if ".key" in push_token:
            with open(push_token, "r") as f:
                push_token = f.read().strip()
        
        print("Preparing to push to hub...")
        print("Token:", push_token)
        push_to_huggingface(test_df, hf_id, push_token, Split.TEST, not args.no_shuffle)
        push_to_huggingface(val_df, hf_id, push_token, Split.VALIDATION, not args.no_shuffle)
        push_to_huggingface(train_df, hf_id, push_token, Split.TRAIN, not args.no_shuffle)

    print(df)

