VIDEO_EXTENSIONS = [
    "mp4",
    "webm"
]

IMAGE_EXTENSIONS  = [
    "gif",
    "jpg",
    "png"
]

if __name__ == "__main__":
    import sys
    import glob
    import os
    import gc
    import pandas as pd
    import argparse
    import shutil
    import PIL
    import pathlib
    from processing.image_processing import get_hash_from_path
    from processing.video_processing import get_video_hash

    p = argparse.ArgumentParser("Process files in various ways")
    p.add_argument("--move_to_folder", type=str, default=None)
    p.add_argument("--delete_files", action="store_true")
    p.add_argument("--rename_to_hash", action="store_true")
    p.add_argument("--copy_to_folder", type=str, default=None)
    p.add_argument("paths", nargs=argparse.REMAINDER)

    args = p.parse_args()
    print(args.paths)
    paths = [p for p in args.paths if "*" not in p]
    print(paths)
    if len(paths) == 0:
        paths = []
        for p in args.paths:
            paths.extend(glob.glob(p))

    paths_df = pd.DataFrame({"full_path": paths})

    if args.move_to_folder != None:
        target_dir = args.move_to_folder if args.move_to_folder.endswith("/") or args.move_to_folder.endswith("\\") else args.move_to_folder + "/"
        target_dir = pathlib.Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        overall_amt = len(paths_df)
        print("Moving Files...")
        for i, row in paths_df.iterrows():
            print(f"Moving: {i + 1}/{overall_amt}", end="\r")
            if not os.path.exists(row["full_path"]):
                continue

            file_name = os.path.split(row["full_path"])[-1]
            new_path = target_dir / file_name
            try:
                shutil.move(row["full_path"], new_path)
            except:
                pass
        
        print(f"Finished moving {overall_amt} files!")
    
    if args.delete_files:
        overall_amt = len(paths_df)
        print("Deleting Files...")
        for i, row in paths_df.iterrows():
            print(f"Deleting: {i + 1}/{overall_amt}", end="\r")
            if not os.path.exists(row["full_path"]):
                continue

            os.remove(row["full_path"])
        
        print(f"Finished deleting {overall_amt} files!")


    

    if args.rename_to_hash:
        overall_amt = len(paths_df)
        print("Renaming Files...")
        for i, row in paths_df.iterrows():
            print(f"Renaming: {i + 1}/{overall_amt}", end="\r")
            if not os.path.exists(row["full_path"]):
                continue
            
            file_name = os.path.split(row["full_path"])[-1]
            ext = file_name.split(".")[-1]

            if ext.lower() in IMAGE_EXTENSIONS:
                try:
                    file_hash, _, _, _ = get_hash_from_path(row["full_path"])
                except PIL.UnidentifiedImageError as e:
                    print(f"Error in file {row["full_path"]}: {e}")
                    os.remove(row["full_path"])
                    print(f"Removed file {row["full_path"]}")
            elif ext.lower() in VIDEO_EXTENSIONS:
                file_hash = get_video_hash(row["full_path"])
                print(file_hash)
                #print(type(file_hash))
            #ext = file_name.split(".")[-1]

            path_segs = [s for s in os.path.split(row["full_path"]) if s != ""]

            new_path = os.path.join(*path_segs[:-1], f"{file_hash}.{ext.lower()}")

            # print(new_path)
            # print(row["full_path"])
            if os.path.exists(row["full_path"]):
                shutil.move(row["full_path"], new_path)
        print(f"Finished renaming {overall_amt} files!")


    if args.copy_to_folder != None:
        target_dir = args.move_to_folder if args.copy_to_folder.endswith("/") or args.copy_to_folder.endswith("\\") else args.copy_to_folder + "/"
        target_dir = pathlib.Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        overall_amt = len(paths_df)
        print("Copying Files...")
        for i, row in paths_df.iterrows():
            print(f"Copying: {i + 1}/{overall_amt}", end="\r")
            if not os.path.exists(row["full_path"]):
                continue

            file_name = os.path.split(row["full_path"])[-1]
            new_path = target_dir / file_name
            if os.path.exists(new_path):
                continue
            try:
                shutil.copy(row["full_path"], new_path)
            except:
                print(f"Failed to copy {row["full_path"]}")
        print(f"Finished copying {overall_amt} files!")