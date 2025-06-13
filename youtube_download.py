# # Utilizes software from https://github.com/yt-dlp/yt-dlp
# import sys
# import os

# def download_youtube(video, extract_audio=False, output_path=None):
#     if "," in video:
#         video = video.split(",")

#     def build_command(url):
#         if extract_audio:
#             base_cmd = f'yt-dlp -x --audio-format wav'
#         else:
#             base_cmd = f'yt-dlp --format mp4'

#         if output_path:
#             base_cmd += f' -o "{output_path}"'

#         return f'{base_cmd} "{url}"'

#     if isinstance(video, str):
#         os.system(build_command(video))
#     else:
#         for i, url in enumerate(video):
#             # Append an index if multiple videos are downloaded to avoid overwriting
#             suffix = f"_{i}" if output_path and len(video) > 1 else ""
#             adjusted_output = (output_path.replace('.mp4', f"{suffix}.mp4")
#                                if output_path and not extract_audio else
#                                output_path.replace('.wav', f"{suffix}.wav")
#                                if output_path else None)
#             os.system(build_command(url).replace(output_path, adjusted_output) if output_path else build_command(url))

# if __name__ == "__main__":
#     args = sys.argv[1:]
#     extract_audio = "-x" in args
#     output_path = None

#     # Remove known flags and capture positional arguments
#     clean_args = []
#     for arg in args:
#         if arg.startswith("-o="):
#             output_path = arg.split("=", 1)[1]
#         elif arg != "-x":
#             clean_args.append(arg)

#     video = ",".join(clean_args)
#     download_youtube(video, extract_audio, output_path)

import os
import subprocess
import argparse

def get_youtube_id(video_url):
    y_id = video_url.split("/")[-1] if "youtu.be" in video_url else video_url.split("v=")[-1] #https://www.youtube.com/watch?v=SC2eSujzrUY https://youtu.be/1UPYKC2FjOw?list=PLRQGRBgN_EnqZ6iBjqDIlTI13SpiYNtmQ
    if "?" in y_id:
        y_id = y_id.split("?")[0]
    if "&" in y_id:
        y_id = y_id.split("&")[0]
    
    return y_id

def download_youtube(video_urls, extract_audio=False, output_path=None):
    results = []

    def build_command(url, out_path=None):
        base_cmd = [
            "yt-dlp",
            "-f", "mp4" if not extract_audio else "bestaudio",
        ]

        if extract_audio:
            base_cmd += ["-x", "--audio-format", "wav"]

        if out_path:
            base_cmd += ["-o", out_path]

        base_cmd.append(url)
        return base_cmd

    def get_metadata(url):
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--print", "%(title)s\n%(uploader)s",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return lines[0], lines[1] if len(lines) >= 2 else "Unknown"
        else:
            return "Unknown Title", "Unknown Uploader"

    for i, url in enumerate(video_urls):
        title, uploader = get_metadata(url)
        suffix = f"_{i}" if output_path and len(video_urls) > 1 else ""
        adjusted_output = None

        if output_path:
            ext = ".wav" if extract_audio else ".mp4"
            if not output_path.endswith(ext):
                output_path += ext
            adjusted_output = output_path.replace(ext, f"{suffix}{ext}")

        cmd = build_command(url, adjusted_output)
        subprocess.run(cmd)
        results.append((title, uploader))

    return results if len(results) > 1 else results[0]

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos or extract audio.")

    parser.add_argument("-x", "--extract-audio", action="store_true", help="Extract audio as WAV")
    parser.add_argument("-o", "--output", type=str, help="Output file path (filename.mp4 or filename.wav)")
    parser.add_argument("urls", nargs=argparse.REMAINDER, help="YouTube video URL(s)")

    args = parser.parse_args()
    info = download_youtube(args.urls, args.extract_audio, args.output)
    print(info)

if __name__ == "__main__":
    main()

