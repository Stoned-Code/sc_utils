from moviepy import VideoFileClip
import librosa
import os
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import gc
import re

from sc_utils.processing.image_processing import is_solid_color, square_padding

import hashlib
from typing import List

def get_audio_hash(y, sr):
    """
    Compute a hash of the audio data.

    Parameters:
    y (numpy.ndarray): Audio time series from librosa's load function.
    sr (int): Sampling rate of the audio.

    Returns:
    str: Hexadecimal string representing the hash of the audio data.
    """
    # Scale y to 16-bit integer range and clip to [-32767, 32767]
    y_scaled = np.clip(y * 32767, -32767, 32767)
    # Round to nearest integer and convert to int16
    y_int16 = np.round(y_scaled).astype(np.int16)
    
    # Convert sr to 4 bytes (little-endian)
    sr_bytes = sr.to_bytes(4, byteorder='little')
    
    # Convert y_int16 to bytes
    y_bytes = y_int16.tobytes()
    
    # Concatenate sr_bytes and y_bytes
    data = sr_bytes + y_bytes
    
    # Compute SHA-256 hash
    hash_object = hashlib.sha256(data)
    hash_hex = hash_object.hexdigest()
    
    return hash_hex

def mel_to_rgb(S,sr=None,is_power=True,ref=np.max,top_db=80.0,cmap='viridis',vmin=None,vmax=None,flip_vertical=False):
    """
    Convert a Mel‐spectrogram to an RGB image.

    Parameters
    ----------
    S           : np.ndarray [shape=(n_mels, t)]
                  Mel‐spectrogram, either power (amplitude squared) or in dB.
    sr          : int or None
                  (unused here, present for API‐compatibility)
    is_power    : bool
                  If True, `S` is assumed to be power spectrogram and
                  will be converted to dB via librosa.power_to_db.
                  If False, `S` is assumed already in dB.
    ref         : scalar or callable
                  Reference power for dB conversion. Passed directly to
                  librosa.power_to_db.
    top_db      : float
                  Threshold the output at `max(S_dB) - top_db`.
    cmap        : str or matplotlib Colormap
                  Colormap name or object for mapping normalized dB → RGB.
    vmin, vmax  : float or None
                  Minimum / maximum dB values for normalization. If None,
                  defaults to [max_dB–top_db, max_dB] when is_power=True,
                  or [min(S), max(S)] when is_power=False.
    flip_vertical : bool
                  If True, flip the image vertically (so low freqs appear at
                  bottom).

    Returns
    -------
    rgb     : np.ndarray [uint8, shape=(n_mels, t, 3)]
              RGB image, with values in [0,255].
    """

    # 1) Convert to dB if needed
    if is_power:
        S_db = librosa.power_to_db(S, ref=ref, top_db=top_db)
        max_db = S_db.max()
        default_vmin = max_db - top_db
        default_vmax = max_db
        vmin = default_vmin if vmin is None else vmin
        vmax = default_vmax if vmax is None else vmax
    else:
        S_db = S
        vmin = S_db.min() if vmin is None else vmin
        vmax = S_db.max() if vmax is None else vmax

    # 2) Normalize to [0,1]
    norm = np.clip((S_db - vmin) / (vmax - vmin), 0.0, 1.0)

    # 3) Apply colormap → RGBA in [0,1]
    cmap = cm.get_cmap(cmap)
    rgba = cmap(norm)             # shape (n_mels, t, 4)

    # 4) Discard alpha, scale to [0,255]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    # 5) Optionally flip vertically
    if flip_vertical:
        rgb = np.flipud(rgb)
    if not is_power:
        del rgba, cmap, norm, vmax, vmin, S_db
    else:
        del rgba, cmap, norm, vmax, vmin, default_vmax, default_vmin, max_db, S_db
    gc.collect()
    return rgb

def multi_square_crop_mel(
    mel: np.ndarray,
    count: int
) -> List[np.ndarray]:
    """
    Crop `count` square patches from a 2D melspectrogram array.
    
    - If the spectrogram is wider than it is tall (time > frequency), 
      the crops run left→right (along the time axis).
    - Otherwise (vertical dominance or square), crops run top→bottom 
      (along the frequency axis).
    - Special case: if count == 1, return the single centered square.
    
    Parameters
    ----------
    mel : np.ndarray
        A 2D array of shape (freq_bins, time_frames).
    count : int
        Number of square crops to produce (must be ≥ 1).
    
    Returns
    -------
    List[np.ndarray]
        A list of length `count`, each a square sub-array of shape (S, S),
        where S = min(freq_bins, time_frames).
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    if mel.ndim != 2:
        raise ValueError("Input mel spectrogram must be a 2D NumPy array.")

    H, W = mel.shape
    S = min(H, W)  # side length of each square
    # total span along the longer axis that can be shifted
    total_offset = (W - S) if W > H else (H - S)

    # Special case: single centered crop
    if count == 1:
        if W > H:
            row_start = 0
            col_start = (W - S) // 2
        else:
            row_start = (H - S) // 2
            col_start = 0
        return [mel[row_start : row_start + S, col_start : col_start + S]]

    # Otherwise, we space `count` windows evenly from 0 to total_offset
    step = total_offset / (count - 1)
    crops: List[np.ndarray] = []
    for i in range(count):
        offset = int(round(i * step))
        if W > H:
            row_start, col_start = 0, offset
        else:
            row_start, col_start = offset, 0
        square = mel[row_start : row_start + S, col_start : col_start + S]
        crops.append(square)

    return crops

def read_audio_from_video(path):
    # Load the video file
    video = VideoFileClip(path)

    # Extract the audio from the video
    audio = video.audio
    if audio == None:
        return None, None
    # Save the audio to a temporary file
    audio.write_audiofile("temp_audio.wav")

    # Load the audio file with librosa
    y, sr = librosa.load("temp_audio.wav")
    os.remove("temp_audio.wav")

    del audio, video
    gc.collect()

    return y, sr


def required_seconds_for_melspec_width(
    width_frames: int,
    sample_rate: int,
    hop_length: int = 512,
    n_fft: int = None
) -> float:
    """
    Calculate how many seconds of audio you need to produce a mel-spectrogram
    with the desired number of time-frames (width).

    Parameters
    ----------
    width_frames : int
        Number of time frames (i.e., the 'width' of the mel-spectrogram).
    sample_rate : int
        Audio sampling rate in Hz (samples per second).
    hop_length : int, optional
        Number of samples between successive frames. Defaults to 512.
    n_fft : int or None, optional
        FFT window size in samples. If provided, the exact time to cover
        the first window is included. If None, this extra padding is ignored.

    Returns
    -------
    float
        Required audio duration in seconds.
    """
    if n_fft is None:
        # Simple approximation: just hop_length * frames
        total_samples = width_frames * hop_length
    else:
        # Exact: first frame lasts n_fft samples, then each additional frame hops
        total_samples = n_fft + (width_frames - 1) * hop_length

    return total_samples / sample_rate


def process_audio(y, sr, mels):
    # Now you can use librosa to analyze the audio data
    # For example, extract MFCCs
    #mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mels)
    mfccs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mels)
    return mfccs


def process_mel(mel, scale, pad_to_square = True, use_grayscale=False):
    if type(mel) == str:
        mel = Image.open(mel)

    # if use_grayscale:
    #     img = img.convert("L")
    # else:
    #     img = img.convert("RGB")
    # width = img.width
    # height = img.height 
    height, width = mel.shape

    if pad_to_square:
        # img = image_processing.pad_to_square(img, use_grayscale)
        # Padding Starts here
        shape = mel.shape

        start_X = 0
        start_y = 0

        if width > height:
            padding = width - height

            padding_top  = int(padding * 0.5)
            padding_bottom = padding - padding_top

            start_X = 0
            start_y = padding_top + 1

            padding_top = np.zeros((padding_top, width, shape[2]))
            padding_bottom = np.zeros((padding_bottom, width, shape[2]))

            concated = np.concat([padding_top, mel, padding_bottom], axis=0)
            shape = mel.shape
            # if not use_grayscale:
            #     return concated.astype(np.uint8), start_X, start_y, width, height
            # else:

            #     if concated.shape[-1] != 1:
            #         concated = np.array(Image.fromarray(concated.astype(np.uint8)).convert("L"))

            mel = concated.reshape((concated.shape[0], concated.shape[1], 1)).astype(np.uint8), start_X, start_y, width, height

        elif height > width:
            padding = height - width
            padding_left = int(padding * 0.5)
            padding_right = padding - padding_left

            start_X = padding_left + 1
            start_y = 0

            padding_left = np.zeros((height, padding_left, shape[2]))
            padding_right = np.zeros((height, padding_right, shape[2]))

            concated = np.concat([padding_left, mel, padding_right], axis = 1)

            # if not use_grayscale:
            #     return concated.astype(np.uint8), start_X, start_y, width, height
            # else:

            #     if concated.shape[-1] != 1:
            #         concated = np.array(Image.fromarray(concated.astype(np.uint8)).convert("L"))
            mel = concated.reshape((concated.shape[0], concated.shape[1], 1)), start_X, start_y, width, height

        #elif height == width:
            
            # if not use_grayscale:
            #     return mel, start_X, start_y, width, height
            # else:
            #     if img.shape[-1] != 1:
            #         img = np.array(Image.fromarray(img.astype(np.uint8)).convert("L"))
            #     return img.reshape((img.shape[0], img.shape[1])).astype(np.uint8), start_X, start_y, width, height
        # Padding Ends Here.
        new_scale = None
        if width > height:
            ratio = height / width
            new_scale = (scale[0], int(scale[1] * ratio))

        elif height > width:
            ratio = width / height
            #print((int(scale[0] * ratio), scale[1]))
            new_scale = (int(scale[0] * ratio), scale[1])
        if new_scale != None:
            mel = mel.resize(new_scale)

        #print(img.width, img.height)
        mel, start_X, start_y, width, height = square_padding(mel, use_grayscale)
        # print(img.shape)
        #mel = Image.fromarray(mel)
        #print(img.shape)
        mel = np.resize(mel, scale)
        
        #mel = np.array(mel)
        #print(img.shape)
        #print(img.shape)
    else:
        mel = np.resize(mel, scale)
        #mel = np.array(mel).astype(np.uint8)

        #img = img
        # img = img.astype(np.uint8)
    # img = Image.fromarray(img)

    #print(img.shape)
    if width != scale[0]:
        width = scale[0]
    
    if height != scale[1]:
        height = scale[1]



    # if invert:
    #     img = 255 - img
    # if normalize:
    #     img = img / 255
    #img = (img - 127.5) / 127.5
    # print(img.dtype)
    #print(img.shape)
    if pad_to_square:
        return mel, start_X, start_y, width, height
    
    return mel, None, None, width, height


if __name__ == "__main__":
    import argparse
    import pathlib
    from sc_utils.processing.image_processing import process_image, closeness_to_black, closeness_to_white, get_hash
    import pandas as pd
    p = argparse.ArgumentParser("Process Audio")
    p.add_argument("--mels", type=int, default=64)
    p.add_argument("--seconds", type=float, default=-1.0)
    p.add_argument("--output_folder", type=str, default="spectogram")
    p.add_argument("--exclude_solids", action="store_true")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args()

    output_folder = args.output_folder
    if not output_folder.endswith("/") or not output_folder.endswith("\\"):
        output_folder = output_folder + "/"
    
    output_folder = pathlib.Path(output_folder)
    #output_folder.mkdir(parents=True, exist_ok=True)
    video_paths = args.rest

    print(video_paths)
    for idx, path in enumerate(video_paths):
        print(f"Processing {idx + 1}")
        audio_folder = os.path.split(path)[-1]
        audio_folder = audio_folder.split(".")[0]
        audio_folder = output_folder / (audio_folder + "/")
        meta_path = audio_folder / "metadata.csv"

        if os.path.exists(meta_path):
            continue

        y, sr = read_audio_from_video(path) if path.split(".")[-1] in ["mp4", "mkv"] else librosa.load(path)
        df = pd.DataFrame(columns=["hash", "file", "blackness", "whiteness", "width", "height", "raw"])
        print(type(y), type(sr))
        if sr != None:
            seconds = args.seconds
            length = sr if seconds == -1.0 else int(seconds * sr)

            # frames_folder = audio_folder / "mels/"
            #audio_folder = pathlib.Path(audio_folder)

            #frames_folder.mkdir(parents=True, exist_ok=True)

            approved = 1

            for i in range(0, len(y) - 1, length):
                y_seg = y[i:i + length]
                print(y_seg.shape)
                if len(y_seg) < length:
                    y_seg = np.concatenate([y_seg, np.zeros((length - len(y_seg)))])

                mfccs = process_audio(y_seg, sr, args.mels)
                #mfccs = mfccs.reshape(*mfccs.shape, 1)
                #mfccs = np.concatenate([mfccs, mfccs, mfccs], axis=2)
                
                mfccs = mel_to_rgb(mfccs, sr=sr)
                # print("Max:", np.max(mfccs))
                # print("Min:", np.min(mfccs))
                # print("Mel Shape:", mfccs.shape)
                mfccs = Image.fromarray(mfccs)
                if args.exclude_solids and is_solid_color(mfccs):
                    continue

                img_hash = get_hash(mfccs)
                file = f"mel_{approved}.jpg"
                blackness = closeness_to_black(mfccs)
                whiteness = closeness_to_white(mfccs)
                df.loc[len(df)] = {"hash": img_hash, "file": file, "blackness": blackness, "whiteness": whiteness, "width": mfccs.width, "height": mfccs.height, "raw":y_seg}
                segment_path = audio_folder / "frames/"
                segment_path.mkdir(parents=True, exist_ok=True)

                segment_path = segment_path / file
                mfccs.save(segment_path)


                # print("Samplerate:", sr)
                # print("Shape:", y.shape)

                approved += 1

        if len(df) > 0:
            df.to_csv(meta_path, index=False)