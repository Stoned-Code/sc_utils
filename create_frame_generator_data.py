# python create_frame_generator_data.py --split_by_parent --blackness_thresh 0.9 --whiteness_thresh 0.9 --seg_amt 3 --output_dir "s:\Data\Frame_Generation\mantis-x_cropped" --meta mantis-x_3_split.csv "s:\Vault\mantis x\video_frames\*\metadata.csv"
import pandas as pd
import glob
import os
import gc
from image_processing import multi_square_crop, process_image
import pathlib
from sc_datasets import shuffle_dataset, split_by_column, balance_by_column
from PIL import Image
import numpy as np
from enum import Enum
import random
from datasets import Dataset
import h5py

class SplitType(Enum):
    PADDING = "padded"
    CROPPING = "cropped"

class DataSplit(Enum):
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

def load_metadata_paths(glob_paths, callback=None):
    paths = []

    for p in glob_paths:
        paths.extend(glob.glob(p))


    current_frame = 0
    overall_frames = 0
    overall_amt = len(paths)
    fp = ""
    dfs = []
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

def create_v_split(output_dir, df, split, size, split_type, use_grayscale, prefix, status_callback):
    def split_status_callback(status):
        if status_callback is not None:
            status_callback(status)
    
    if type(size) == int:
        size = (size, size)

    # Set up output directory and file path
    if isinstance(output_dir, str):
        path = Path(output_dir if output_dir.endswith("/") or output_dir.endswith("\\") else output_dir + "/")
        path.mkdir(parents=True, exist_ok=True)
    else:
        path = output_dir
    h5file = path / (f"{prefix}_{size[0]}x{size[1]}_{split.value}.h5")
    nfiles = len(df)
    split_status_callback(f"There are {nfiles} files for {h5file}")

    # Open HDF5 file and create datasets with optimized settings
    with h5py.File(h5file, "w") as h5f:
        split_status_callback("Creating X dataset...")
        channels = 3 if not use_grayscale else 1
        X_ds = h5f.create_dataset(
            "X",
            shape=(nfiles, *size, channels),
            dtype=np.uint8,
            chunks=(1, *size, channels)
        )
        split_status_callback("Creating y dataset...")
        y_ds = h5f.create_dataset(
            "y",
            shape=(nfiles, *size, channels),
            dtype=np.uint8,
            chunks=(1, *size, channels)
        )

        # Process each row in the DataFrame
        for cnt, irow in df.iterrows():
            seg_count = irow["segment_count"]
            seg = irow["segment"] if irow["segment"] != -1 else 0

            # Open and process images
            raw_X_img = Image.open(irow["path"])
            raw_y_img = Image.open(irow["y_path"])
            if split_type == SplitType.CROPPING:
                X_imgs = multi_square_crop(raw_X_img, seg_count)
                y_imgs = multi_square_crop(raw_y_img, seg_count)  # Fixed: was using raw_X_img
            elif split_type == SplitType.PADDING:
                if seg_count > 1:
                    X_imgs = split_image(raw_X_img, seg_count)
                    y_imgs = split_image(raw_y_img, seg_count)
                else:
                    X_imgs = [raw_X_img]
                    y_imgs = [raw_y_img]
            
            X = X_imgs[seg]
            y = y_imgs[seg]

            # Process images (ensure process_image returns np.uint8 arrays)
            X, _, _, _, _ = process_image(X, size, use_grayscale, False, True, False)
            y, _, _, _, _ = process_image(y, size, use_grayscale, False, True, False)

            # Write to datasets with error handling
            try:
                X_ds[cnt] = X.astype(np.uint8)
                y_ds[cnt] = y.astype(np.uint8)
            except RuntimeError as e:
                split_status_callback(f"Error writing to dataset at index {cnt}: {e}")
                raise

            split_status_callback(f"Processing {split.value} data: {cnt + 1}/{nfiles}")

            # Flush every 100 iterations
            if (cnt + 1) % 100 == 0:
                h5f.flush()
                split_status_callback(f"Flushed at {cnt + 1}")

            # Clean up
            del raw_X_img, raw_y_img, X_imgs, y_imgs
            gc.collect()

                    
def get_segment_length(amt, overall_amt):
    seg_len = overall_amt / amt
    if seg_len % 1 != 0:
        seg_len = int(seg_len) + 1
    else:
        seg_len = int(seg_len)
    
    return seg_len


def main(_df, scale, output_dir, prefix="frame_gen", segment_amt = -1, shuffle=True, segment_thresh = 1, segment_offset = 1, lower_segment_amt = 2, 
    use_grayscale=False, split_type = SplitType.CROPPING, split_by_parent=True, meta_path = None, test_ratio = 0.2, bal_by_column=False, balance_col="parent", col_trim = -1, callback_status=None):
    
    def init_status_callback(status):
        if callback_status != None:
            callback_status(status)

    if not output_dir.endswith("/") and not output_dir.endswith("\\"):
        output_dir += "/"
    
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Max Column Balance:", col_trim)
    init_status_callback("Created output path...")
    X, y = [], []

    overall_amt = len(_df)

    # print(_df.columns)
    #train_path = output_dir / f"{suffix}_{scale}x{scale}_{suffix}.h5"

    if os.path.exists(meta_path):
        save_df = pd.read_csv(meta_path)
    else:
    # elif meta_path == None or not os.path.exists(meta_path):
        save_df = pd.DataFrame(columns=["path", "y_path", "segment", "segment_count", "parent"])

        i_real = 0
        for i, row in _df.iterrows():  
            ratio = max([row["width"], row["height"]]) / min([row["width"], row["height"]])
            if segment_amt > 0:
                segment_count = segment_amt
            else:
                if split_type == SplitType.CROPPING:
                    if ratio > 1 and ratio <= segment_thresh:
                        segment_count = lower_segment_amt
                    elif ratio > segment_thresh:
                        segment_count = int(ratio + segment_offset) if ratio % 1 <= 0.5 else int(ratio + segment_offset) + 1
                    elif ratio == 1:
                        segment_count == 1

            if split_type == SplitType.CROPPING:
                #print("Segment Count:", segment_count)
                for j in range(segment_count):
                    save_df.loc[len(save_df)] = {
                        "path": row["path"],
                        "y_path": row["y_path"],
                        "segment": j,
                        "segment_count": segment_count,
                        "parent": row["parent"]
                    }
                

            else:
                save_df.loc[len(save_df)] = {
                    "path": row["path"],
                    "segment": -1,
                    "segment_count": 1,
                    "parent": row["parent"]
                }
            init_status_callback(f"Retrieving Segments: {i_real + 1}/{overall_amt}")
            i_real += 1
        save_df = save_df.sample(frac=1)
        save_df.reset_index(drop=True, inplace=True)
        if meta_path != None:
            save_df.to_csv(meta_path, index=False)

    if shuffle:
        save_df = save_df.sample(frac=1)

    overall_amt = len(save_df)
    accepted = 0
    save_df["exists"] = False
    save_df.reset_index(drop=True, inplace=True)
    for i, row in save_df.iterrows():
        if os.path.exists(row["path"]) and os.path.exists(row["y_path"]):
            save_df.at[i, "exists"] = True
            accepted += 1
        init_status_callback(f"Checking Exists: {i + 1}/{overall_amt}, Accepted: {accepted}")

    save_df = save_df[save_df["exists"] == True]

    if bal_by_column:
        save_df = balance_by_column(save_df, balance_col, trim=col_trim)

    if split_by_parent:
        train_df, val_df, test_df = split_by_column(save_df, "parent", test_ratio)
    else:
        train_df = save_df.sample(frac=1.0 - test_ratio)
        val_df = save_df.drop(train_df.index)
        test_df = val_df.sample(frac=0.5)
        val_df = val_df.drop(test_df.index)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    create_v_split(output_dir, test_df, DataSplit.TEST, scale, split_type, use_grayscale, prefix, init_status_callback)

    create_v_split(output_dir, val_df, DataSplit.VALIDATION, scale, split_type, use_grayscale, prefix, init_status_callback)

    create_v_split(output_dir, train_df, DataSplit.TRAIN, scale, split_type, use_grayscale, prefix, init_status_callback)


def push_to_huggingface(df, huggingface_id, push_token, split: DataSplit, shuffle=True):
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
    p.add_argument("--prefix", type=str, default="frame_gen")
    p.add_argument("--seg_amt", type=int, default = 0)
    p.add_argument("--seg_offset", type=int, default=1)
    p.add_argument("--seg_thresh", type = float, default=1)
    p.add_argument("--lower_seg_amt", type=int, default = 2)
    p.add_argument("--scale", type=int, default = 240)
    p.add_argument("--output_dir", type=str, default="./data")
    p.add_argument("--test_ratio", type=float, default = 0.2)
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--padding", action="store_true")
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--normalize", action = "store_true")
    p.add_argument("--push_token", type=str, default=None)
    p.add_argument("--split_by_parent", action="store_true")
    p.add_argument("--hf_id", type=str, default=None)
    p.add_argument("--meta", type=str, default = None)
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
    # print("Creating training split...")
    # train_df = df.sample(frac=1 - args.test_ratio)
    # print("Creating test split...")
    # test_df = df.drop(train_df.index)
    # print("Creating validation split...")
    # val_df = test_df.sample(frac=0.5)
    # test_df = test_df.drop(val_df.index)
    # print("Finished splitting data frames!")
    # train_df.reset_index(drop=True, inplace=True)
    # test_df.reset_index(drop=True, inplace=True)
    # val_df.reset_index(drop=True, inplace=True)
    split_type = SplitType.CROPPING if not args.padding else SplitType.PADDING

    main(df, args.scale, args.output_dir, args.prefix, args.seg_amt, not args.no_shuffle, args.seg_thresh, args.seg_offset, args.lower_seg_amt, args.grayscale, split_type, args.split_by_parent,
    args.meta, args.test_ratio, print)

    # if args.push_token == None:
    #     print("No push token, creating numpy array.")
    #     segment_amt = get_segment_count(args.segment_len, len(train_df)) if args.segment_len > 0 else args.segment_len
    #     save_dataset(test_df, (args.scale, args.scale), args.output_dir, args.prefix, "test", segment_amt, -1, not args.no_shuffle, not args.no_padding, args.crop_amt, 
    #                 args.crop_thresh, args.crop_offset, 2, args.normalize, args.invert, args.grayscale)
        
    #     save_dataset(val_df, (args.scale, args.scale), args.output_dir, args.prefix, "val", segment_amt, -1, not args.no_shuffle, not args.no_padding, args.crop_amt,
    #                 args.crop_thresh, args.crop_offset, 2, args.normalize, args.invert, args.grayscale)

    #     save_dataset(train_df, (args.scale, args.scale), args.output_dir, args.prefix, "train", segment_amt, -1, not args.no_shuffle, not args.no_padding, args.crop_amt,
    #                 args.crop_thresh, args.crop_offset, 2, args.normalize, args.invert, args.grayscale)
    # else:
    #     assert args.push_token != None and args.hf_id != None

    #     push_token = args.push_token
    #     hf_id = args.hf_id

    #     if ".key" in push_token:
    #         with open(push_token, "r") as f:
    #             push_token = f.read().strip()
        
    #     print("Preparing to push to hub...")
    #     print("Token:", push_token)
    #     push_to_huggingface(test_df, hf_id, push_token, DataSplit.TEST, not args.no_shuffle)
    #     push_to_huggingface(val_df, hf_id, push_token, DataSplit.VALIDATION, not args.no_shuffle)
    #     push_to_huggingface(train_df, hf_id, push_token, DataSplit.TRAIN, not args.no_shuffle)

    # print(df)

