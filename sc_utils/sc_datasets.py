import numpy as np
import pandas as pd
import glob
import h5py
import gc
import json
from PyPDF2 import PdfReader

def balance_by_column(df, column, reset_index=False, trim = -1):
    col_unique = df[column].unique()
    item_amounts_df = {"item": [], "amount": []}
    item_amounts_df = pd.DataFrame(item_amounts_df)
    print("Trim:", trim)
    if trim < 0:
        for col in col_unique:
            temp_df = df[df[column] == col]
            item_amounts_df.loc[len(item_amounts_df)] = {"item": col, "amount": len(temp_df)}
        

        minimum = item_amounts_df["amount"].min() if trim < 0 else trim
    else:
        minimum = trim
    print("Minimum:", minimum)
    new_df = []

    for col in col_unique:
        temp_df = df[df[column] == col]
        temp_df = temp_df.sample(n=minimum if len(temp_df) > minimum else len(temp_df))

        new_df.append(temp_df)
    
    new_df = pd.concat(new_df)
    new_df = new_df.sample(frac=1)
    if reset_index:
        new_df.reset_index(drop=True, inplace=True)
    return new_df

def merge_dataframes(df_paths):
    df = [pd.read_csv(p) for p in df_paths]
    df = pd.concat(df)
    df.reset_index(drop=True, inplace=True)
    return df


def numpy_to_h5(np_paths, output_path, use_y=False):
    # paths = []
    # #print(np_paths)
    # for path in np_paths:
    #     if "*" in path:
    #         paths.extend(glob.glob(path))
    #     else:
    #         paths.append(path)
    
    count = 0
    channels = 0
    size = None
    for p in np_paths:
        temp_np = np.load(p)
        count += temp_np.shape[0]
        if p == np_paths[-1]:
            channels = temp_np.shape[-1] if len(temp_np.shape) == 4 else 1
            size = temp_np.shape[1:3]
            #print("Size:", size)
        del temp_np
        gc.collect()

    with h5py.File(output_path, "w") as h5f:
        print(size)
        X_ds = h5f.create_dataset(
            "X",
            shape=(count, *size, channels),
            dtype=np.uint8,
            chunks = (1, *size, channels)
        )

        if use_y:
            y_ds = h5f.create_dataset(
                "y",
                shape=(count, *size, channels),
                dtype=np.uint8,
                chunks = (1, *size, channels)
            )

        idx = 0
        for p in np_paths:
            temp_np = np.load(p)
            if use_y:
                temp_np_y = np.load(p.replace("_X_", "_y_"))
                for x, y in zip(temp_np, temp_np_y):
                    X_ds[idx] = x.astype(np.uint8)
                    y_ds[idx] = y.astype(np.uint8)

                    idx += 1
                    print(f"Processing: {idx}/{count}", end="\r")
                
                del temp_np, temp_np_y
            else:
                for a in temp_np:
                    #print(a.shape)
                    X_ds[idx] = a.astype(np.uint8)
                    idx += 1
                    print(f"Processing: {idx}/{count}", end="\r")

                del temp_np
            if idx % 100 == 0:
                h5f.flush()
            gc.collect()


    print(f"Saved combined data to {output_path}")


def shuffle_dataset(X, y = None):
    keys = np.random.permutation(X.shape[0])
    X = X[keys]
    if type(y) != type(None):
        y = y[keys]
        return X, y

    return X


def split_by_column(df, col, test_ratio = 0.2):
    unique = df[col].unique()

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    for u in unique:
        temp_df = df[df[col] == u]
        temp_df.reset_index(drop=True, inplace=True)
        
        temp_train = temp_df.sample(frac= 1 - test_ratio)
        temp_val = temp_df.drop(temp_train.index)
        temp_test = temp_val.sample(frac=0.5)
        temp_val = temp_val.drop(temp_test.index)
        if len(temp_val) == 0 or len(temp_test) == 0 or len(temp_val) == 0:
            print("Unique:", u, "Length:", len(temp_df))
            continue
        train_df = pd.concat([train_df, temp_train])
        val_df = pd.concat([val_df, temp_val])
        test_df = pd.concat([test_df, temp_test])

    
    return train_df, val_df, test_df

def create_unstructured_jsonl(paths, output_path, encoding="utf-8"):
    path_count = len(paths)
    with open(output_path, "w", encoding=encoding) as of:
        for i, path in enumerate(paths):
            print(f"Reading {i + 1}/{path_count}", end="\r")

            if path.split(".")[-1] == "txt":
                with open(path, "r", encoding=encoding) as ipf:
                    text = ipf.read()
                    data = {"text": text.strip()}

                    of.write(json.dumps(data) + "\n")
            elif path.split(".")[-1] == "pdf":
                reader = PdfReader(path)

                text = ""
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                
                data = {"text": text.strip()}

                of.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    #p.add_argument("df_paths", nargs=argparse.REMAINDER)
    p.add_argument("--merge_df", action="store_true")
    p.add_argument("--npy_to_h5", action="store_true")
    p.add_argument("--output_path", required=True, type=str)
    p.add_argument("--use_y", action="store_true")
    p.add_argument("--create_unstructured_text_data", action="store_true")
    p.add_argument("paths", nargs=argparse.REMAINDER)

    args = p.parse_args()

    paths = []

    for path in args.paths:
        if "*" in path:
            paths.extend(glob.glob(path))
        else:
            paths.append(path)


    if args.merge_df:
        df = merge_dataframes(paths)
        df.to_csv(args.output_path, index=False)
    
    elif args.npy_to_h5:
        print(paths)
        numpy_to_h5(paths, args.output_path, args.use_y)

    elif args.create_unstructured_text_data:
        create_unstructured_jsonl(paths, args.output_path)
