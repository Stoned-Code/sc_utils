import numpy as np
import pandas as pd


def balance_by_column(df, column, reset_index=False, trim = -1):
    col_unique = df[column].unique()
    item_amounts_df = {"item": [], "amount": []}
    item_amounts_df = pd.DataFrame(item_amounts_df)

    for col in col_unique:
        temp_df = df[df[column] == col]
        item_amounts_df.loc[len(item_amounts_df)] = {"item": col, "amount": len(temp_df)}
    
    minimum = item_amounts_df["amount"].min() if trim < 0 else trim
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

        train_df = pd.concat([train_df, temp_train])
        val_df = pd.concat([val_df, temp_val])
        test_df = pd.concat([test_df, temp_test])

    
    return train_df, val_df, test_df

if __name__ == "__main__":
    import os
    path = "output_metadata.csv"

    df = pd.read_csv(path)
    df["dataset"] = df["full_path"].apply(lambda fp: os.path.split(fp)[-2].split("/")[-1])
    print(df.head())
    print("Pre Length:", len(df))
    df = balance_by_column(df, "dataset")
    print(df.head())
    print("Post Length:", len(df))