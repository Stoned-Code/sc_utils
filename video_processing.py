import imageio.v3 as iio
import pathlib
from PIL import Image
import pandas as pd
import os
import sys
import gc
import glob
from image_processing import get_hash, is_solid_color, closeness_to_black, closeness_to_white, set_shortest_length, get_sim_hash, ssim
import numpy as np
import hashlib
import shutil
import cv2

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open the video file")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_video_hash(file_path):
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as file:
        chunk_size = 4096
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

# # Path to your MKV file
# def process_video_frames(video_path, output_path = "frame_output/", omit_solid = False, omit_similar=False):
#     output_path = pathlib.Path(output_path)
#     df = pd.DataFrame(columns=["hash", "path", "blackness", "whiteness", "width", "height"])
#     frame_path = output_path / "frames/"
#     frame_path.mkdir(exist_ok=True, parents=True)
#     meta_path = os.path.join(output_path, "metadata.csv")
#     if not os.path.exists(meta_path):
#         approved = 0
#         # Read the video frame by frame
#         print("Attempting to process", video_path)
#         for idx, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
#             # Process each frame
#             print(f"Processing frame {approved}/{idx} \"{video_path}\"", end = "\r")
#             img_path = f"{frame_path}/frame_{idx:03d}.jpg"
#             img = Image.fromarray(frame)
#             if is_solid_color(img) and omit_solid:
#                 del img
#                 gc.collect()
#                 continue
        
#             img_hash, width, height, ratio = get_hash(img)
#             img_blackness = closeness_to_black(img)
#             img_whiteness = closeness_to_white(img)
#             #print(type(frame))
#             # Example: Save the frame as an image
#             hashes = sorted(df["hash"].values)
#             #print(type(hashes))
#             if omit_similar:
#                 if img_hash not in hashes:
#                     approved += 1
#                     iio.imwrite(img_path, frame)
#                     df.loc[len(df)] = {"hash": img_hash, "path": os.path.split(img_path)[-1], "blackness": img_blackness, "whiteness": img_whiteness, "width": width, "height": height}
#             else:
#                 approved += 1
#                 iio.imwrite(img_path, frame)
#                 df.loc[len(df)] = {"hash": img_hash, "path": os.path.split(img_path)[-1], "blackness": img_blackness, "whiteness": img_whiteness, "width": width, "height": height}            
#             del img_hash, width, height, ratio, hashes, img
#             gc.collect()

#         print(f"Processing frame {approved}/{idx} {video_path}")
#         print(len(df))
#         #df = df.drop_duplicates(subset=["hash"])
#         df.to_csv(meta_path, index=False)
#         print(len(df))

#         del df
#         gc.collect()

def process_video_frames(video_path, output_path="frame_output/", omit_solid=False, omit_similar=False, set_shortest_len = -1, callback=None, delete_errored=False, hash_size = 64, use_similarity_hash=True):
    output_path = pathlib.Path(output_path)
    df = pd.DataFrame(columns=["hash", "file", "blackness", "whiteness", "width", "height", "exists"])
    frame_path = output_path / "frames/"
    frame_path.mkdir(exist_ok=True, parents=True)
    meta_path = os.path.join(output_path, "metadata.csv")
    
    if not os.path.exists(meta_path):
        approved = 0
        # Check if the file is a GIF or a video
        if video_path.lower().endswith(".gif"):
            print("Attempting to process GIF", video_path)
            # Read the GIF frame by frame using Pillow
            with Image.open(video_path) as img:
                idx = 0
                while True:
                    try:
                        img.seek(idx)
                        frame = img.copy().convert('RGB')
                        if callback != None:
                            callback(approved, idx, video_path)
                        print(f"Processing frame {approved}/{idx} \"{video_path}\"", end="\r")
                        img_hash, width, height, ratio = get_hash(frame) if not use_similarity_hash else get_sim_hash(frame, hash_size)
                        img_blackness = closeness_to_black(frame)
                        img_whiteness = closeness_to_white(frame)            
                        if is_solid_color(frame) and omit_solid:
                            df.loc[len(df)] = {
                                "hash": img_hash, 
                                "file": os.path.split(img_path)[-1], 
                                "blackness": img_blackness, 
                                "whiteness": img_whiteness, 
                                "width": frame.width, 
                                "height": 
                                frame.height, 
                                "exists": os.path.exists(img_path)
                                }
                            idx += 1
                            continue


                        if set_shortest_len > 0:
                            frame = np.array(set_shortest_length(frame, set_shortest_len))

                        img_path = f"{frame_path}/frame_{idx:03d}.jpg"
                        hashes = sorted(df["hash"].values)
                        if type(frame) != Image:
                            frame = Image.fromarray(frame)
                        frame = frame.convert('RGB')
                        if omit_similar:
                            if img_hash not in hashes:
                                approved += 1
                                     
                                frame.save(img_path)
                                #df.loc[len(df)] = {"hash": img_hash, "file": os.path.split(img_path)[-1], "blackness": img_blackness, "whiteness": img_whiteness, "width": frame.width, "height": frame.height}
                        else:
                            approved += 1
                            frame.save(img_path)
                        
                        df.loc[len(df)] = {
                            "hash": img_hash, 
                            "file": os.path.split(img_path)[-1], 
                            "blackness": img_blackness, 
                            "whiteness": img_whiteness, 
                            "width": frame.width, 
                            "height": 
                            frame.height, 
                            "exists": os.path.exists(img_path)
                            }
                        
                        idx += 1
                    except EOFError:
                        break
                    except Image.DecompressionBombError:
                        break
        else:
            # Read the video frame by frame using imageio
            print("Attempting to process", video_path)
            try:
                for idx, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
                    if callback != None:
                        callback(approved, idx, video_path)
                    print(f"Processing frame {approved}/{idx} \"{video_path}\"", end="\r")
                    img_path = f"{frame_path}/frame_{idx:03d}.jpg"
                    img = Image.fromarray(frame)
                    img_hash, width, height, ratio = get_hash(img)
                    img_blackness = closeness_to_black(img)
                    img_whiteness = closeness_to_white(img)

                    if type(frame) == np.ndarray:
                        #print("Shape:", frame.shape)
                        h, w, _ = frame.shape
                    else:
                        w, h = frame.width, frame.height

                    if is_solid_color(img) and omit_solid:
                        del img
                        df.loc[len(df)] = {
                            "hash": img_hash, 
                            "file": os.path.split(img_path)[-1], 
                            "blackness": img_blackness, 
                            "whiteness": img_whiteness, 
                            "width": w, 
                            "height": h, 
                            "exists": os.path.exists(img_path)
                            }
                        idx += 1
                        gc.collect()
                        continue
                    

                    
                    hashes = sorted(df["hash"].values)
                    if set_shortest_len > 0:
                        frame = np.array(set_shortest_length(img, set_shortest_len))

                    if omit_similar:
                        if img_hash not in hashes:
                            approved += 1
                            iio.imwrite(img_path, frame)
                            #df.loc[len(df)] = {"hash": img_hash, "file": os.path.split(img_path)[-1], "blackness": img_blackness, "whiteness": img_whiteness, "width": frame.shape[1], "height": frame.shape[0]}
                    else:
                        approved += 1
                        iio.imwrite(img_path, frame)

                    df.loc[len(df)] = {
                        "hash": img_hash, 
                        "file": os.path.split(img_path)[-1], 
                        "blackness": img_blackness, 
                        "whiteness": img_whiteness, 
                        "width": frame.shape[1], 
                        "height": frame.shape[0], 
                        "exists": os.path.exists(img_path)}
                    
                    del img_hash, width, height, ratio, hashes, img
                    gc.collect()
            except Exception as ex:
                shutil.rmtree(output_path)
                if delete_errored:
                    os.remove(video_path)
                
                raise ex
                #return
        print(f"Processing frame {approved}/{idx} {video_path}")
        print(len(df))
        df.to_csv(meta_path, index=False)
        print(len(df))

        del df
        gc.collect()

def process_videos_frames(video_paths, output_path = "frame_output/", omit_solid=False, omit_similar=False, set_shortest_len = -1, hash_size = 64, use_similarity_hash=True, callback=None):
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    overall_len = len(video_paths)
    for vid_idx, video_path in enumerate(video_paths):

        
        print(f"Processing {vid_idx}/{overall_len}: \"{video_path}\"")
        ext = video_path.split(".")[-1]

        video_output = output_path / os.path.split(video_path)[-1].replace(f".{ext}", "/")
        process_video_frames(video_path, str(video_output), omit_solid, omit_similar, set_shortest_len, callback, False, hash_size, use_similarity_hash)

if __name__ == "__main__":
    video_paths = sys.argv[-1]
    #print(video_paths)
    if "," in video_paths:
        temp_paths = video_paths.split(",")
        video_paths = []
        for vp in temp_paths:
            paths = glob.glob(vp)
            video_paths.extend(paths)
        
        del temp_paths
        gc.collect()

    else:
        video_paths = glob.glob(video_paths)
    
    omit_solid = "--omit_solid" in sys.argv
    omit_similar = "--omit_similar" in sys.argv
    set_shortest_len = int(sys.argv[sys.argv.index("--set_shortest") + 1]) if "--set_shortest" in sys.argv else -1

    print("Video Count:", len(video_paths))
    output_path = "frame_output/" if "-O" not in sys.argv else sys.argv[sys.argv.index("-O") + 1]

    if not output_path.endswith("/"):
        output_path = output_path + "/"
    # print(video_paths)
    process_videos_frames(video_paths, output_path, omit_solid, omit_similar, set_shortest_len)