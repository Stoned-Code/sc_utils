import glob # Imports glob for path management.
import shutil # Import shutil for handling files.
import tqdm

VIDEO_EXTENSIONS = [
    "mp4",
    "webm",
    "avi",
    "mkv",
    "mov",
    "flv",
    "wmv",
    "mpeg",
    "mpg",
    "3gp"
]

IMAGE_EXTENSIONS = [
    "gif",
    "jpg",
    "png",
    "jpeg",
    "bmp",
    "tiff",
    "webp",
    "heic",
    "svg",
    "ico"
]

def delete_empty_folders(paths_df, path_col="full_path"):
    """
    Delete any folders that are empty.
    
    :paths_df: A dataframe of paths
    """
    # Grabs directories.
    directories_df = paths_df[paths_df[path_col].apply(lambda p: os.path.isdir(p))]
    # Creates an empty list.
    files_removed = []

    overall_amt = len(directories_df)

    # Loops through the directories.
    for i, (idc, row) in enumerate(directories_df.iterrows()):
        # Grabs the path.
        paths = os.listdir(row[path_col])
        
        if len(paths) == 0:
            # Deletes the directory if it's empty.
            shutil.rmtree(row[path_col])
            # Adds the path to "files_removed"
            files_removed.append(row[path_col])

            print(f"Removed path: {i + 1}/{overall_amt}", end="\r")
    
    print(f"Deleted {len(files_removed)} directories.")

if __name__ == "__main__":
    import os
    import pandas as pd
    import argparse
    import PIL
    import pathlib
    from processing.image_processing import get_hash_from_path
    from processing.video_processing import get_video_hash

    p = argparse.ArgumentParser("Process files in various ways")

    p.add_argument("--move_to_folder", help="Specify a folder to move all paths to.", type=str, default=None)
    p.add_argument("--delete_files", help="Deletes all parsed paths.", action="store_true")
    p.add_argument("--delete_folders", help="Deletes all parsed folders.", action="store_true")
    p.add_argument("--rename_to_hash", help="Renames listed files to their hashes.", action="store_true")
    p.add_argument("--copy_to_folder", help="Specify a directory to copy the paths to.", type=str, default=None)
    p.add_argument("--delete_empty_folders", help="Deletes all folders among the paths and deletes them if they're empty.", action="store_true")
    p.add_argument("--delete_existing", help="If flag is active it deletes existing files with hash...", action="store_true")
    p.add_argument("paths", nargs=argparse.REMAINDER)

    args = p.parse_args()

    print(f"There are {len(args.paths)} inputed paths")
    
    paths = []

    # Loops through remainding argument paths.
    for p in args.paths:
        if "*" in p:
            # If "*" is in an argument path, it extends the list with a list of globbed paths.
            paths.extend(glob.glob(p))
        elif os.path.exists(p):
            # Adds the path to the list if it exists.
            paths.append(p)
    
    print(f"There are {len(paths)} total paths")

    # Creates a pandas dataframe from the paths.
    paths_df = pd.DataFrame({"full_path": paths})
    
    # Grabs the amount of paths in the dataframe
    overall_amt = len(paths_df)

    if args.delete_empty_folders:
        # Deletes folders that are empty.
        delete_empty_folders(paths_df)

    if args.delete_folders:
        temp_df = paths_df[paths_df["full_path"].apply(lambda p: os.path.exists(p) and os.path.isdir(p))]

        temp_df.reset_index(drop=True, inplace=True)

        for i, row in tqdm.tqdm(temp_df.iterrows(), total=len(temp_df), desc="Removing File"):
            print(f"Removing file {i + 1}/{len(temp_df)}", end="\r")
            path = row["full_path"]
            shutil.rmtree(path)
        
        print()

    if args.move_to_folder != None:
        # Turns target folder into a pathlib path.
        target_dir = args.move_to_folder if args.move_to_folder.endswith("/") or args.move_to_folder.endswith("\\") else args.move_to_folder + "/"
        target_dir = pathlib.Path(target_dir)

        # Creates the directory if it doesn't exist.
        target_dir.mkdir(parents=True, exist_ok=True)

        print("Moving Files...")
        
        # Iterrates through the rows of the dataframe.
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=len(paths_df), desc="Moving"):
            
            # print(f"Moving: {i + 1}/{overall_amt}", end="\r")
            
            # Grabs the filename from the path.
            file_name = os.path.split(row["full_path"])[-1]
            # Creates target path.
            new_path = target_dir / file_name
            
            try:
                # Tries to move the path to the new one.
                shutil.move(row["full_path"], new_path)
            except:
                pass
        
        print(f"Finished moving {overall_amt} files!")
    
    if args.delete_files:
        print("Deleting Files...")
        
        # Iterrates through the rows of the dataframe.
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=len(paths_df), desc="Deleting"):
            
            # print(f"Deleting: {i + 1}/{overall_amt}", end="\r")
            
            if not os.path.exists(row["full_path"]):
                continue
            
            # Removes the path.
            os.remove(row["full_path"])
        
        print(f"Finished deleting {overall_amt} files!")

    if args.rename_to_hash:
        print("Renaming Files...")
        
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=len(paths_df), desc="Renaming"):
            if not os.path.exists(row["full_path"]):
                print("Skipping file:", row["full_path"])
                continue
            
            # print(f"Renaming: {i + 1}/{overall_amt}", end="\r")
            
            # Grabs filesname.
            file_name = os.path.split(row["full_path"])[-1]
            # Grfabs the files extension.
            ext = file_name.split(".")[-1]

            if ext.lower() in IMAGE_EXTENSIONS:
                try:
                    # Gets the hash of image from it's path.
                    file_hash, _, _, _ = get_hash_from_path(row["full_path"])
                    # print(file_hash)
                except PIL.UnidentifiedImageError as e:
                    print(f"Error in file {row["full_path"]}: {e}")
                    # Removes image if this error is 
                    os.remove(row["full_path"])
                    
                    print(f"Removed file {row["full_path"]}")
                except PermissionError as e:
                    os.remove(row["full_path"])
                    print(f"Permission error, deleting file {row["full_path"]}")
                
                except FileNotFoundError as e:
                    print(f"File not found: \"{row["full_path"]}\"")
            elif ext.lower() in VIDEO_EXTENSIONS:
                file_hash = get_video_hash(row["full_path"])
                #print(file_hash)
                #print(type(file_hash))
            
            # Segments the path by directory depth.
            path_segs = [s for s in os.path.split(row["full_path"]) if s != ""]
            # print("File Hash:", file_hash)
            # Creates a new path with the files hash.
            new_path = os.path.join(*path_segs[:-1], f"{file_hash}.{ext.lower()}")
            # print("New Path:", new_path)
            # Skips over renaming the file if it's already named that.
            if os.path.exists(new_path) and args.delete_existing and not os.path.samefile(new_path, row["full_path"]):#new_path.replace("/", "\\") != row["full_path"].replace("/", "\\"):
                # print("File already exists, skipping:", new_path)
                os.remove(row["full_path"])
                print(f"Deleted file wish existing hash: \"{row["full_path"]}\"")
                continue

            elif os.path.exists(new_path) and not args.delete_existing:
                continue

            if os.path.exists(row["full_path"]):
                try:
                    # Renames the path.
                    shutil.move(row["full_path"], new_path)
                except FileExistsError as ex:
                    print()
                    print(f"Can't find specified path: {row["full_path"]}")
                except FileNotFoundError as ex:
                    print()
                    print(f"Can't find specified path: {row["full_path"]}")        

        print(f"Finished renaming {overall_amt} files!")


    if args.copy_to_folder != None:
        # Turns target directory into a pathlib path.
        target_dir = args.move_to_folder if args.copy_to_folder.endswith("/") or args.copy_to_folder.endswith("\\") else args.copy_to_folder + "/"
        target_dir = pathlib.Path(target_dir)

        # Creates the directory if it doesn't exist.
        target_dir.mkdir(parents=True, exist_ok=True)

        print("Copying Files...")
        
        for i, row in tqdm.tqdm(paths_df.iterrows(), total=len(paths_df), desc="Copying"):
            # print(f"Copying: {i + 1}/{overall_amt}", end="\r")
            # Skips over the row if it doesn't exist.
            if not os.path.exists(row["full_path"]):
                continue
            
            # Grabs the files name.
            file_name = os.path.split(row["full_path"])[-1]
            
            # Creates the new path.
            new_path = target_dir / file_name
            # Skips over the path if it already exists.
            if os.path.exists(new_path):
                continue
            
            try:
                # Tries to copy the file over to the new directory.
                shutil.copy(row["full_path"], new_path)
            except:
                print(f"Failed to copy {row["full_path"]}")
        print(f"Finished copying {overall_amt} files!")