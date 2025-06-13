from PIL import Image, ImageFile
import cv2
import glob
import pandas as pd
import sys
import numpy as np
import gc
import pathlib
from image_processing import square_padding, process_image, is_solid_color, get_hash, get_hash_from_path, multi_square_crop
from audio_processing import multi_square_crop_mel, process_mel
import os
from sc_datasets import shuffle_dataset

def to_numpy(df, scale, use_grayscale, invert, suffix, segments = 1, pad_to_square=True, output_folder = "data",
prefix="autoencoder", path_column="full_path", crop_thresh = -1, crop_segments = -1, crop_offset=1, shuffle=True, lower_thresh_crop_segments=2,
verbose=True, callback_save=None, callback_row_complete=None, read_from_np=True):
    X = []
    #y = []
    amt = len(df)
    if not os.path.exists(output_folder):
        temp_path = output_folder if not output_folder.endswith("/") or not output_folder.endswith("\\") else output_folder + "/"
        temp_path = pathlib.Path(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        del temp_path
        gc.collect()
    #outputs = {l: i for i, l in enumerate(sorted(df[label_col].unique()))}
    if segments != 1:
        segments = 1 / segments
        segments = int(amt * segments) + 1
        print("Segment Length:", segments)
    segment = 1

    text_spacing = " " * 20
    overall_amt = 0
    
    for i, row in df.iterrows():
        
        full_path = row[path_column]
        # print(f"{i + 1}/{amt} {full_path} {text_spacing}", end = "\n" if i == amt - 1 else "\r")
        if verbose:
            print(f"{i + 1}/{amt} {full_path}", end= " ")
        else:
            print(f"{i + 1}/{amt}", end=" " if crop_thresh > 1 or crop_segments > 1 else "\r")
        if not read_from_np:
            img = Image.open(row[path_column])
            
            w = img.width
            h = img.height
        else:
            img = np.load(row[path_column])
            print("Image Shape:", img.shape)
            if len(img.shape) == 3:
                h, w, _ = img.shape
            else:
                h, w = img.shape

        ratio = max([w, h]) / min([w,h])
        if crop_thresh > 1 or crop_segments > 1:
            # if (crop_thresh > 1 and ratio > crop_thresh) or crop_segments > 1:
            if ratio > crop_thresh:
                crop_segs = int(ratio + crop_offset) if crop_segments <= 1 else crop_segments
                try:
                    if not read_from_np:
                        crop_squares = multi_square_crop(img, crop_segs) 
                    else:
                        crop_squares =  multi_square_crop_mel(img, crop_segs)
                except:
                    continue
                
                print("Crop Amount:", len(crop_squares), text_spacing, end = "\n" if i == amt - 1 else "\r")
                for crop_seg in crop_squares:
                    if pad_to_square:
                        if not read_from_np:
                            crop_seg, start_X, start_y, width, height = process_image(crop_seg, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                        else:
                            crop_seg, start_X, start_y, width, height = process_mel(crop_seg, scale, pad_to_square)
                        #seg_label = [start_X, start_y, width, height]
                    else:
                        if not read_from_np:
                            crop_seg, _, _, width, height = process_image(crop_seg, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                        else:
                            crop_seg, start_X, start_y, width, height = process_mel(crop_seg, scale, pad_to_square)
                        #seg_label = [width, height]

                    X.append(crop_seg)
                    #y.append(seg_label)
                    overall_amt += 1

                    if segments != 1 and overall_amt % segments == 0 and overall_amt != 0:
                        prev_shape = None
                        for x in X:
                            if prev_shape != None and prev_shape != x.shape:
                                print(f"Wrong Shape: {x.shape}")
                            prev_shape = x.shape
                        X = np.array(X)
                        #y = np.array(y)
                        
                        if shuffle:
                            X = shuffle_dataset(X)
                        x_save_path = f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy"
                        np.save(x_save_path, X)
                        if callback_save != None:
                            callback_save(x_save_path, segment)
                        print(f"Saved segment: \"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy\"{text_spacing}")
                        #np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_y_{segment}.npy", y)
                        segment += 1

                        del X
                        gc.collect()

                        X = []
                        #y = []
                if pad_to_square:
                    del crop_segs, crop_seg, crop_squares, start_X, start_y, width, height
                else:
                    del crop_segs, crop_seg, crop_squares, width, height
                gc.collect()

            elif ratio > 1 and ratio < crop_thresh and crop_thresh > 1:
                crop_segs = lower_thresh_crop_segments
                crop_squares = list(multi_square_crop(img, crop_segs))
                print("Crop Amount:", len(crop_squares), text_spacing, end = "\n" if i == amt - 1 else "\r")
                for crop_seg in crop_squares:
                    if pad_to_square:
                        if not read_from_np:
                            crop_seg, start_X, start_y, width, height = process_image(crop_seg, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                        else:
                            crop_seg, start_X, start_y, width, height = process_mel(crop_seg, scale, pad_to_square)
                        #seg_label = [start_X, start_y, width, height]
                    else:
                        if not read_from_np:
                            crop_seg, _, _, width, height = process_image(crop_seg, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                        else:
                            crop_seg, start_X, start_y, width, height = process_mel(crop_seg, scale, pad_to_square)

                    X.append(crop_seg)
                    #y.append(seg_label)
                    overall_amt += 1

                    if segments != 1 and overall_amt % segments == 0 and overall_amt != 0:
                        prev_shape = None
                        for x in X:
                            if prev_shape != None and prev_shape != x.shape:
                                print(f"Wrong Shape: {x.shape}")
                            prev_shape = x.shape
                        X = np.array(X)
                        #y = np.array(y)
                        
                        if shuffle:
                            X = shuffle_dataset(X)
                        np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy", X)

                        print(f"Saved segment: \"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy\"{text_spacing}")
                        #np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_y_{segment}.npy", y)
                        segment += 1

                        del X
                        gc.collect()

                        X = []
                        #y = []
                if pad_to_square:
                    del crop_segs, crop_seg, crop_squares, start_X, start_y, width, height
                else:
                    del crop_segs, crop_seg, crop_squares, width, height
                gc.collect()
            else:
                # if pad_to_square:
                #     img, start_X, start_y, width, height  = process_image(img, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                #     label = [ start_X, start_y, width, height ] # Label, Start X, Start y, Width, Height
                # else:
                #     img, _, _, width, height  = process_image(img, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                #     label = [ width, height ] # Label, Start X, Start y, Width, Height
                if pad_to_square:
                    if not read_from_np:
                        img, start_X, start_y, width, height = process_image(img, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                    else:
                        img, start_X, start_y, width, height = process_mel(img, scale, pad_to_square)
                    #seg_label = [start_X, start_y, width, height]
                else:
                    if not read_from_np:
                        img, _, _, width, height = process_image(img, scale, use_grayscale, invert, pad_to_square=pad_to_square)
                    else:
                        img, start_X, start_y, width, height = process_mel(img, scale, pad_to_square)
                X.append(img)
                #y.append(label)
                overall_amt += 1

                if segments != 1 and overall_amt % segments == 0 and overall_amt != 0:
                    prev_shape = None
                    for x in X:
                        if prev_shape != None and prev_shape != x.shape:
                            print(f"Wrong Shape: {x.shape}")
                        prev_shape = x.shape
                    X = np.array(X)
                    #y = np.array(y)
                    
                    if shuffle:
                        X = shuffle_dataset(X)
                    np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy", X)

                    print(f"Saved segment: \"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy\"{text_spacing}")
                    #np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_y_{segment}.npy", y)
                    segment += 1

                    del X
                    gc.collect()

                    X = []
                    y = []

        else:
        #try:
            img, start_X, start_y, width, height  = process_image(img, scale, use_grayscale, invert, pad_to_square=pad_to_square)
            
            
            
            # except:
            #     img = process_image(row["path"], scale, use_grayscale, pad_to_square=False)


            label = [ start_X, start_y, width, height ] # Label, Start X, Start y, Width, Height
            #label = 1 if row["Like"] == True else 0

            X.append(img)
            #y.append(label)
            overall_amt += 1

            if segments != 1 and overall_amt % segments == 0 and overall_amt != 0:
                prev_shape = None
                for x in X:
                    if prev_shape != None and prev_shape != x.shape:
                        print(f"Wrong Shape: {x.shape}")
                    prev_shape = x.shape
                X = np.array(X)
                #y = np.array(y)
                
                if shuffle:
                    X = shuffle_dataset(X)
                np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy", X)
                # spacing = " " * 100
                print(f"Saved segment: \"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy\"{text_spacing}")
                #np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_y_{segment}.npy", y)
                segment += 1

                del X
                gc.collect()

                X = []
                y = []
        if callback_row_complete != None:
            callback_row_complete(i + 1, amt, segment)

    X = np.array(X)
    #y = np.array(y)
    if X.shape[0] > 0:
        if shuffle:
            X = shuffle_dataset(X)
        np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X.npy" if segments == 1 else f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy", X)
        #np.save(f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_y.npy" if segments == 1 else f"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_y_{segment}.npy", y)
        print(f"Saved segment: \"{output_folder}/{prefix}_{scale[0]}x{scale[1]}_{suffix}_X_{segment}.npy\"{text_spacing}")
    
    gc.collect()
    return X


def get_metadata(path, file_column = "file"):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["hash"])
    if "full_path" not in df.columns:
        df["full_path"] = df[file_column].apply(lambda p: os.path.join(*os.path.split(path)[:-1], "frames", p))
    
    df["parent"] = os.path.split(path)[-2]
    
    return df


def get_glob_metadata(path, file_column = "file"):
    paths = glob.glob(path)
    #print(paths)
    print("Found", len(paths), "files...")
    dfs = []
    for path in paths:
        df = get_metadata(path, file_column)

        dfs.append(df)
    
    df = pd.concat(dfs)
    
    df = df.drop_duplicates(subset=["hash"])
    return df


def filter_solid(df, path_col="full_path", callback=None):
    new_df = pd.DataFrame(columns=df.columns)
    all_df_amt = len(df)
    for i, row in df.iterrows():
        if callback != None:
            callback(len(new_df), i + 1, all_df_amt)
        print(f"Filtering solid images: {i + 1}/{all_df_amt}", end="\r")
        img = Image.open(row[path_col])
        if not is_solid_color(img):
            new_df.loc[len(new_df)] = row
        img.close()
        del img
        gc.collect()
    print(f"Filtering solid images: {i + 1}/{all_df_amt}")
    return new_df


def filter_hash(df, path_col="full_path", callback = None):
    if "hash" not in df.columns:
        new_df = pd.DataFrame(columns=list(df.columns) + ["hash"])
        print(new_df.columns)
        all_df_amt = len(df)

        for i, row in df.iterrows():
            if callback != None:
                callback(len(new_df), i + 1, all_df_amt)
            print(f"Filtering images by hash: {len(new_df)}/{i + 1}/{all_df_amt}", end="\r")
            hashes = new_df["hash"].values
            img_hash, _, _, _ = get_hash_from_path(row[path_col])
            if img_hash not in hashes:
                new_df.loc[len(new_df)] = {path_col:row[path_col], "hash": img_hash}
        del hashes, _, df
        print(f"Filtering images by hash: {len(new_df)}{i + 1}/{all_df_amt}")
        gc.collect()
        return new_df
    else:
        #new_df = pd.DataFrame(columns = df.columns)
        new_df = df.drop_duplicates(subset=["hash"])
        new_df.reset_index(drop=True, inplace=True)

    
    return new_df



if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = sys.argv[-1]
    use_hash_filter = "--filter_hash" in sys.argv
    use_solid_filter = "--filter_solid" in sys.argv
    filter_existing = "--filter_exists" in sys.argv
    use_meta = "--use_meta" in sys.argv
    autoencoder_prefix = "autoencoder" if "--prefix" not in sys.argv else sys.argv[sys.argv.index("--prefix") + 1]
    crop_threshold = -1 if "--crop_thresh" not in sys.argv else float(sys.argv[sys.argv.index("--crop_thresh") + 1])
    crop_lower_thresh = 2 if "--crop_lower_thresh_amount" not in sys.argv else int(sys.argv[sys.argv.index("--crop_lower_thresh_amount") + 1])
    crop_segments = -1 if "--crop_segments" not in sys.argv else int(sys.argv[sys.argv.index("--crop_segments") + 1])
    crop_offset = 1 if "--crop_offset" not in sys.argv else int(sys.argv[sys.argv.index("--crop_offset") + 1])
    path_column = "full_path" if "--path_column" not in sys.argv else sys.argv[sys.argv.index("--path_column") + 1]
    maximum_overall = -1 if "--maximum" not in sys.argv else int(sys.argv[sys.argv.index("--maximum") + 1])
    verbose = "-v" in sys.argv
    shuffle = "--shuffle" in sys.argv
    meta_path = None if "--meta_path" not in sys.argv else sys.argv[sys.argv.index("--meta_path") + 1]


    if "," in data:
        _data = data.split(",")
        data = []
        for m in _data:
            data.extend(glob.glob(m))
    if not use_meta:
        df = pd.DataFrame({path_column: data})
    else:
        print("Using Meta...")
        df = get_glob_metadata(data)

    if use_hash_filter:
        df = filter_hash(df)
    
    if use_solid_filter:
        df = filter_solid(df)

    if filter_existing:
        df = df[df[path_column].apply(lambda p: os.path.exists(p))]

    if meta_path != None:
        df.to_csv(meta_path, index=False)

    if maximum_overall == -1:
        df = df.sample(frac=1)
    else:
        df = df.sample(n=maximum_overall)
    df.reset_index(inplace=True, drop=True)
    print(df.head())
    print(len(df))

    test_ratio = 0.2
    print("Creating training split...")
    train_df = df.sample(frac=1 - test_ratio)
    print("Creating test split...")
    test_df = df.drop(train_df.index)
    print("Creating validation split...")
    val_df = test_df.sample(frac=0.5)
    test_df = test_df.drop(val_df.index)
    print("Finished splitting data frames!")
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # test_split = 0.2
    # test_split = int(len(df) * test_split)

    # test_df = df.sample(n=test_split)
    # train_df = df.drop(test_df.index)

    # train_df = train_df.sample(frac=1)
    # test_df = test_df.sample(frac=1)

    print(train_df.head())
    print("Train Length:", len(train_df))
    print(test_df.head())
    print("Test Length:", len(test_df))


    max_seg_length = 5_000 if "--max_segment_length" not in sys.argv else int(sys.argv[sys.argv.index("--max_segment_length") + 1])
    output_folder = "data" if "-O" not in sys.argv else sys.argv[sys.argv.index("-O") + 1]
    output_folder_path = pathlib.Path(output_folder if output_folder.endswith("/") else output_folder + "/")
    output_folder_path.mkdir(parents=True, exist_ok=True)
    seg_amount = 1
    seg_length = int(len(train_df) / seg_amount)

    while seg_length > max_seg_length:
        seg_amount += 1
        seg_length = int(len(train_df) / seg_amount)

    print("Estimated Segment Amount:", seg_amount if crop_segments <= 1 else seg_amount * crop_segments)

    scale = 224 if "--scale" not in sys.argv else int(sys.argv[sys.argv.index("--scale") + 1])
    invert = "--invert" in sys.argv
    grayscale = "--grayscale" in sys.argv
    pad_to_square = "--stretch" not in sys.argv
    #test_seg_amount = int(len(test_df) / seg_amount)
    test_df.reset_index(inplace=True, drop=True)
    X, y = to_numpy(test_df, (scale, scale), grayscale, invert, "test", seg_amount, pad_to_square, output_folder,
     autoencoder_prefix, path_column, crop_threshold, crop_segments, crop_offset, shuffle, crop_lower_thresh, verbose)

    del X, y
    gc.collect()

    print("Estimated Segment Amount:", seg_amount if crop_segments <= 1 else seg_amount * crop_segments)
    val_df.reset_index(inplace=True, drop=True)
    X, y = to_numpy(val_df, (scale, scale), grayscale, invert, "val", seg_amount, pad_to_square, output_folder,
    autoencoder_prefix, path_column, crop_threshold, crop_segments, crop_offset, shuffle, crop_lower_thresh, verbose)
    
    del X, y
    gc.collect()

    print("Estimated Segment Amount:", seg_amount if crop_segments <= 1 else seg_amount * crop_segments)
    train_df.reset_index(inplace=True, drop=True)
    X, y = to_numpy(train_df, (scale, scale), grayscale, invert, "train", seg_amount, pad_to_square, output_folder, 
    autoencoder_prefix, path_column, crop_threshold, crop_segments, crop_offset, shuffle, crop_lower_thresh, verbose)
