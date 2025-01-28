import subprocess
import os
import argparse
from tqdm import tqdm
import re
from datetime import timedelta
import json
import random
import requests
import time
import tempfile
import logging
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import shutil
from faster_whisper import WhisperModel
import sys
import platform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define the VideoProfile class with add_subtitles option
class VideoProfile:
    def __init__(self, add_subtitles, profile_name, audio_viz_config, subtitle_style, use_pexels,
                 intro_video, outro_video, background_music, pexels_keywords, pexels_api_key, youtube_upload):
        self.add_subtitles = add_subtitles
        self.profile_name = profile_name
        self.audio_viz_config = audio_viz_config
        self.subtitle_style = subtitle_style
        self.use_pexels = use_pexels
        self.intro_video = intro_video
        self.outro_video = outro_video
        self.background_music = background_music
        self.pexels_keywords = pexels_keywords
        self.pexels_api_key = pexels_api_key
        self.youtube_upload = youtube_upload


def upload_thumbnail(youtube, video_id, thumbnail_path):
    if not os.path.exists(thumbnail_path):
        logging.error(f"Thumbnail file not found: {thumbnail_path}")
        return False

    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
        ).execute()
        logging.info(f"Thumbnail uploaded successfully for video ID: {video_id}")
        return True
    except Exception as e:
        logging.error(f"An error occurred while uploading the thumbnail: {str(e)}")
        return False

def upload_to_youtube(video_file, title, description, tags, category, privacy_status, credentials_info, publish_at, thumbnail_path=None):
    try:
        credentials = Credentials.from_authorized_user_info(credentials_info)
        youtube = build('youtube', 'v3', credentials=credentials)

        # Convert publish_at to datetime if it's a string
        if isinstance(publish_at, str):
            publish_at = datetime.fromisoformat(publish_at.rstrip('Z'))

        request_body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category
            },
            'status': {
                'privacyStatus': privacy_status,
                'publishAt': publish_at.isoformat() + 'Z'  # Ensure proper UTC format
            }
        }

        logging.info(f"Uploading video with settings:")
        logging.info(f"Title: {title}")
        logging.info(f"Scheduled Publish Time: {publish_at.isoformat()}Z")

        media = MediaFileUpload(video_file, chunksize=-1, resumable=True)

        request = youtube.videos().insert(
            part='snippet,status',
            body=request_body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%")

        video_id = response['id']
        logging.info(f"Video uploaded successfully! Video ID: {video_id}")

        if thumbnail_path:
            thumbnail_success = upload_thumbnail(youtube, video_id, thumbnail_path)
            if not thumbnail_success:
                logging.warning("Video uploaded successfully, but thumbnail upload failed.")

        return video_id
    except Exception as e:
        logging.error(f"An error occurred while uploading the video: {str(e)}")
        raise


def download_pexels_video(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        with open(output_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        logging.info(f"Successfully downloaded video to {output_path}")
    except requests.RequestException as e:
        logging.error(f"Error downloading video from {url}: {str(e)}")
        raise


def get_pexels_videos(keywords, total_duration, pexels_api_key):
    videos = []
    current_duration = 0
    per_page = 80  # Increased to get more videos per request
    max_retries = 3

    while current_duration < total_duration:
        keyword = random.choice(keywords)
        page = random.randint(1, 10)  # Randomly select a page between 1 and 5
        url = f"https://api.pexels.com/videos/search?query={keyword}&per_page={per_page}&page={page}&orientation=landscape&size=medium"
        headers = {"Authorization": pexels_api_key}

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(
                        f"Failed to fetch videos for keyword '{keyword}' after {max_retries} attempts: {str(e)}")
                    break
                logging.warning(f"Attempt {attempt + 1} failed for keyword '{keyword}'. Retrying...")
                continue

        if not data.get('videos'):
            logging.warning(f"No videos found for keyword '{keyword}'. Moving to next search.")
            continue

        # Filter videos to only include horizontal ones
        horizontal_videos = [video for video in data['videos'] if video['width'] > video['height']]

        if not horizontal_videos:
            logging.warning(f"No horizontal videos found for keyword '{keyword}'. Moving to next search.")
            continue

        # Shuffle the filtered videos
        random.shuffle(horizontal_videos)

        for video in horizontal_videos:
            if current_duration >= total_duration:
                break

            video_files = video['video_files']
            hd_video = next((v for v in video_files if v['quality'] == 'hd' and v['width'] > v['height']), None)
            if hd_video:
                videos.append({
                    'url': hd_video['link'],
                    'duration': video['duration']
                })
                current_duration += video['duration']
                logging.info(f"Added video with keyword '{keyword}': duration {video['duration']}s, total duration: {current_duration}s")
                break  # Break after adding one video for this keyword

    return videos


def create_background_video(pexels_videos, total_duration, output_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        video_list_file = os.path.join(temp_dir, 'video_list.txt')
        normalized_videos = []
        
        # First pass: Download and normalize all videos
        for i, video in enumerate(pexels_videos):
            temp_path = os.path.join(temp_dir, f'temp_video_{i}.mp4')
            normalized_path = os.path.join(temp_dir, f'normalized_{i}.mp4')
            
            try:
                # Download video
                download_pexels_video(video['url'], temp_path)
                
                # Check if the downloaded file exists and has content
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    logging.error(f"Downloaded video {i} is empty or missing")
                    continue
                
                # Normalize video
                normalize_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_path,
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p2',
                    '-b:v', '5M',
                    '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                    '-an',
                    '-f', 'mp4',  # Force MP4 format
                    normalized_path
                ]
                result = subprocess.run(normalize_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.error(f"Error normalizing video {i}: {result.stderr}")
                    continue
                
                # Verify the normalized video
                if os.path.exists(normalized_path) and os.path.getsize(normalized_path) > 0:
                    normalized_videos.append(normalized_path)
                    logging.info(f"Successfully processed video {i}")
                else:
                    logging.error(f"Normalized video {i} is empty or missing")
                
                # Clean up temporary download
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                logging.error(f"Error processing video {i}: {str(e)}")
                continue
        
        if not normalized_videos:
            raise RuntimeError("No videos were successfully processed")
        
        # Write the list of successfully processed videos
        with open(video_list_file, 'w', encoding='utf-8') as f:
            for video_path in normalized_videos:
                f.write(f"file '{video_path}'\n")
        
        # Concatenate the videos with a more robust approach
        try:
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', video_list_file,
                '-c:v', 'h264_nvenc',  # Re-encode instead of copy to ensure consistency
                '-preset', 'p2',
                '-b:v', '10M',
                '-t', str(total_duration),
                '-an',
                output_path
            ]
            
            run_ffmpeg_with_progress(ffmpeg_command, total_duration, "Creating Background Video")
            
            # Verify the final output
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Final video file is empty or missing")
                
        except Exception as e:
            logging.error(f"Error during final video concatenation: {str(e)}")
            raise


def create_background_video_from_images(scene_images, scene_durations, output_path):
    # Create a simple filter for concatenation
    inputs = []
    
    # Prepare inputs
    for image, duration in zip(scene_images, scene_durations):
        inputs.extend(['-loop', '1', '-t', f'{duration/1000}', '-i', image])
    
    # Build the concat filter
    concat_inputs = ''.join(f'[{i}:v]' for i in range(len(scene_images)))
    filter_complex = f'{concat_inputs}concat=n={len(scene_images)}:v=1[outv]'
    
    # Build the final command
    ffmpeg_command = [
        'ffmpeg', '-y', '-hwaccel', 'cuda'
    ] + inputs + [
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-c:v', 'h264_nvenc',
        '-preset', 'p2',
        '-b:v', '10M',
        output_path
    ]

    total_duration = sum(duration/1000 for duration in scene_durations)
    run_ffmpeg_with_progress(ffmpeg_command, total_duration, "Creating Background Video")
    
    logging.info("Background video creation completed.")


def check_subtitles_file(subtitles_file):
    if not os.path.exists(subtitles_file):
        raise FileNotFoundError(f"Subtitles file not found: {subtitles_file}")

    # Check if the file is empty
    if os.path.getsize(subtitles_file) == 0:
        raise ValueError(f"Subtitles file is empty: {subtitles_file}")


def generate_subtitles(audio_file, subtitles_file, subtitle_style):
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    try:
        # Transcribe the audio
        logging.info("Starting transcription...")
        segments, info = model.transcribe(audio_file, beam_size=5, language='en')
        logging.info("Transcription completed")

        font_size = subtitle_style.get('font_size', '56')
        primary_color = subtitle_style.get('primary_color', '&H00FFFFFF')
        outline_color = subtitle_style.get('outline_color', '&H00000000')
        back_color = subtitle_style.get('back_color', '&H7F000000')
        bold = subtitle_style.get('bold', '-1')
        italic = subtitle_style.get('italic', '0')
        alignment = subtitle_style.get('alignment', '2')
        margin_v = subtitle_style.get('margin_v', '50')
        margin_h = subtitle_style.get('margin_h', '200')

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{primary_color},{primary_color},{outline_color},{back_color},{bold},{italic},0,0,100,100,0,0,1,2,0,{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        def ass_timestamp(seconds):
            td = timedelta(seconds=seconds)
            hours = int(td.total_seconds() // 3600)
            minutes = int((td.total_seconds() % 3600) // 60)
            seconds = int(td.total_seconds() % 60)
            milliseconds = int((td.total_seconds() - int(td.total_seconds())) * 100)
            return f"{hours:01d}:{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

        ass_events = ""
        for segment in segments:
            start = ass_timestamp(segment.start)
            end = ass_timestamp(segment.end)
            text = segment.text.strip().replace('\n', '\\N')
            ass_events += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"

        ass_content = ass_header + ass_events

        with open(subtitles_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        print(f"Subtitles generated and saved to {subtitles_file}")
    except Exception as e:
        logging.error(f"Error generating subtitles: {str(e)}")
        raise


def run_ffmpeg_with_progress(command, total_duration, label):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    progress_regex = re.compile(r'time=(\d{2}):(\d{2}):(\d{2})\.\d{2}')
    error_output = []

    # Define a default bar format
    default_bar_format = '{l_bar}{bar} | {n:.2f}/{total:.2f}s [{elapsed}<{remaining}]'

    try:
        with tqdm(total=total_duration, unit='sec',
                  bar_format=default_bar_format,
                  desc=label) as pbar:
            for line in process.stdout:
                error_output.append(line)
                matches = progress_regex.search(line)
                if matches:
                    hours, minutes, seconds = map(int, matches.groups())
                    current_time = hours * 3600 + minutes * 60 + seconds
                    pbar.update(current_time - pbar.n)
    except Exception as e:
        logging.error(f"Error in progress bar: {str(e)}")
        # Continue processing without the progress bar

    process.wait()
    if process.returncode != 0:
        print("FFmpeg Error Output:")
        print("".join(error_output[-20:]))  # Print the last 20 lines of output
        raise subprocess.CalledProcessError(process.returncode, command)


def create_simple_video(image_path, audio_file, output_path):
    """
    Creates a video from a single image and audio file.
    The video duration matches the audio duration.
    """
    # Get audio duration
    audio_duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_file
    ]).strip())

    # Create video from image and audio
    command = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-loop', '1',                     # Loop the image
        '-i', image_path,                 # Input 1: Image
        '-i', audio_file,                 # Input 2: Audio
        '-c:v', 'h264_nvenc',            # Use NVIDIA encoder
        '-preset', 'p2',                  # Encoding preset
        '-tune', 'hq',                    # High quality tuning
        '-rc', 'vbr',                     # Variable bitrate
        '-cq', '23',                      # Constant quality factor
        '-b:v', '10M',                    # Video bitrate
        '-maxrate', '15M',                # Maximum bitrate
        '-bufsize', '30M',                # Buffer size
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30',
        '-r', '30',                       # 30 fps
        '-c:a', 'aac',                    # Audio codec
        '-b:a', '320k',                   # Audio bitrate
        '-filter:a', 'volume=0.9',  # Set volume to 50%
        '-ar', '48000',                   # Audio sample rate
        '-ac', '2',                       # 2 audio channels
        '-shortest',                      # Match audio duration
        '-t', str(audio_duration),        # Explicitly set duration to match audio
        output_path
    ]

    run_ffmpeg_with_progress(command, audio_duration, "Creating Simple Video")

def create_video(args, profile, temp_dir, output_dir):
    os.makedirs(f'{output_dir}/Videos', exist_ok=True)

    # Get main audio duration
    audio_duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        args.final_audio_file
    ]).strip())
    
    total_duration = audio_duration + 10
    logging.info(f"Audio duration: {audio_duration} seconds")
    logging.info(f"Total duration with 10s padding: {total_duration} seconds")

    final_video = f'{output_dir}/Videos/FinalVideo.mp4'

    if profile.background_music and os.path.exists(profile.background_music):
        # Create temporary file for mixed audio
        mixed_audio = os.path.join(temp_dir, 'mixed_audio.wav')
        
        # Mix the audio files - main audio at 100% volume, background at 20%
        ffmpeg_mix = [
            'ffmpeg', '-y',
            '-i', args.final_audio_file,
            '-i', profile.background_music,
            '-filter_complex', '[0:a]volume=1.0[a1];[1:a]volume=0.2,aloop=loop=-1:size=0[a2];[a1][a2]amix=inputs=2:duration=first[aout]',
            '-map', '[aout]',
            mixed_audio
        ]
        subprocess.run(ffmpeg_mix, check=True)
        
        # Use mixed audio in video creation
        audio_input = mixed_audio
    else:
        audio_input = args.final_audio_file

    # Create video with the appropriate audio
    command = [
        'ffmpeg', '-y',
        '-loop', '1',
        '-framerate', '1',
        '-i', args.scene_images[0],
        '-i', audio_input,
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'stillimage',
        '-c:a', 'aac',
        '-b:a', '320k',
        '-shortest',
        '-t', str(total_duration),
        final_video
    ]
    
    run_ffmpeg_with_progress(command, total_duration, "Creating Video")

    # Clean up temporary mixed audio file if it was created
    if profile.background_music and os.path.exists(mixed_audio):
        os.remove(mixed_audio)

    # Verify final duration
    final_duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        final_video
    ]).strip())
    logging.info(f"Final video duration: {final_duration} seconds")


def create_local_video(args, profile, output_dir):
    """Optimized function for local video generation on both Windows and Mac"""
    print("Starting local video generation...")
    os.makedirs(f'{output_dir}/Videos', exist_ok=True)

    # Get main audio duration
    audio_duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        args.final_audio_file
    ]).strip())
    
    total_duration = audio_duration + 10
    print(f"Audio duration: {audio_duration} seconds")
    print(f"Total duration with 10s padding: {total_duration} seconds")

    final_video = f'{output_dir}/Videos/FinalVideo.mp4'

    # Handle background music if present
    if profile.background_music and os.path.exists(profile.background_music):
        # Create temporary file for mixed audio
        mixed_audio = os.path.join(output_dir, 'mixed_audio.wav')
        
        # Mix audio with platform-optimized settings
        ffmpeg_mix = [
            'ffmpeg', '-y',
            '-i', args.final_audio_file,
            '-i', profile.background_music,
            '-filter_complex', '[0:a]volume=1.0[a1];[1:a]volume=0.12,aloop=loop=-1:size=0[a2];[a1][a2]amix=inputs=2:duration=first[aout]',
            '-map', '[aout]',
            mixed_audio
        ]
        subprocess.run(ffmpeg_mix, check=True)
        audio_input = mixed_audio
    else:
        audio_input = args.final_audio_file

    # Check input image format and dimensions
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,pix_fmt',
        '-of', 'json',
        args.scene_images[0]
    ]
    
    probe_result = json.loads(subprocess.check_output(probe_cmd).decode('utf-8'))
    stream_info = probe_result['streams'][0]
    
    # Determine if we need to rescale or pad the image
    needs_processing = (
        stream_info['width'] != 1920 or
        stream_info['height'] != 1080 or
        stream_info.get('pix_fmt') != 'yuv420p'  # Most compatible pixel format
    )

    # Determine optimal video settings based on platform
    system = platform.system().lower()
    if system == 'darwin':  # macOS
        hw_accel = []  # No hardware acceleration by default on Mac
        if needs_processing:
            vcodec = ['-c:v', 'h264']  # Only encode if necessary
        else:
            vcodec = ['-c:v', 'copy']  # Direct stream copy if possible
    elif system == 'windows':
        has_nvidia = shutil.which('nvidia-smi') is not None
        hw_accel = ['-hwaccel', 'cuda'] if has_nvidia else []
        if needs_processing:
            vcodec = ['-c:v', 'h264_nvenc'] if has_nvidia else ['-c:v', 'h264']
        else:
            vcodec = ['-c:v', 'copy']
    else:  # Linux or other
        has_nvidia = shutil.which('nvidia-smi') is not None
        hw_accel = ['-hwaccel', 'cuda'] if has_nvidia else []
        if needs_processing:
            vcodec = ['-c:v', 'h264_nvenc'] if has_nvidia else ['-c:v', 'h264']
        else:
            vcodec = ['-c:v', 'copy']

    # Build the video processing command
    command = [
        'ffmpeg', '-y'
    ] + hw_accel + [
        '-loop', '1',
        '-framerate', '30',
        '-i', args.scene_images[0],
        '-i', audio_input
    ]

    # Add video processing only if needed
    if needs_processing:
        command.extend([
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2'
        ])
    
    command.extend(vcodec + [
        '-c:a', 'aac',
        '-b:a', '320k',
        '-shortest',
        '-t', str(total_duration),
        '-movflags', '+faststart',  # Optimize for web playback
        final_video
    ])
    
    print("Running FFmpeg command with the following settings:")
    print(f"Hardware Acceleration: {'Yes' if hw_accel else 'No'}")
    print(f"Video Processing Required: {'Yes' if needs_processing else 'No'}")
    print(f"Video Codec: {vcodec}")
    
    run_ffmpeg_with_progress(command, total_duration, "Creating Local Video")

    # Clean up temporary mixed audio file if it was created
    if profile.background_music and os.path.exists(mixed_audio):
        os.remove(mixed_audio)

    # Add YouTube upload handling
    if isinstance(profile.youtube_upload, dict) and profile.youtube_upload.get('enabled', False):
        logging.info("Starting YouTube upload process...")
        try:
            # Extract all YouTube upload settings
            thumbnail_path = f'{args.temp_dir}/thumbnail.png'
            title = args.title
            description = profile.youtube_upload.get('description', '')
            tags = profile.youtube_upload.get('tags', [])
            category = profile.youtube_upload.get('category', '22')
            privacy_status = profile.youtube_upload.get('privacy_status', 'private')
            
            # Get the next scheduled upload date
            next_upload_date_str = profile.youtube_upload.get('next_upload_date')
            if not next_upload_date_str:
                logging.error("No upload date provided in YouTube settings")
                raise ValueError("Missing next_upload_date in YouTube settings")

            video_id = upload_to_youtube(
                video_file=final_video,
                title=title,
                description=description,
                tags=tags,
                category=category,
                privacy_status=privacy_status,
                credentials_info=json.loads(args.youtube_credentials),
                publish_at=next_upload_date_str,
                thumbnail_path=thumbnail_path
            )

            if video_id:
                logging.info(f"Video successfully scheduled for upload to YouTube. Video ID: {video_id}")
                logging.info(f"Scheduled publish time: {next_upload_date_str}")
            else:
                logging.error("Failed to upload video to YouTube - no video ID returned")
                
        except Exception as e:
            logging.error(f"Failed to upload video to YouTube: {str(e)}", exc_info=True)
            raise

    return final_video

def main(args):
    try:
        start_time = time.time()
        logging.info("Starting video generation process...")
        
        # Log the received arguments
        logging.info("Received arguments:")
        logging.info(f"Profile name: {args.profile_name}")
        logging.info(f"Number of scene images: {len(args.scene_images)}")
        logging.info(f"Number of scene durations: {len(args.scene_durations)}")
        logging.info(f"Temp directory: {args.temp_dir}")
        
        # Create VideoProfile first to check use_pexels flag
        youtube_upload_settings = json.loads(args.youtube_upload)
        if not isinstance(youtube_upload_settings, dict):
            youtube_upload_settings = {'enabled': bool(youtube_upload_settings)}
        youtube_credentials = json.loads(args.youtube_credentials)
        
        logging.info("Creating VideoProfile object...")
        profile = VideoProfile(
            add_subtitles=args.add_subtitles,
            profile_name=args.profile_name,
            audio_viz_config=json.loads(args.audio_viz_config),
            subtitle_style=json.loads(args.subtitle_style),
            use_pexels=args.use_pexels,
            intro_video=args.intro_video if args.intro_video != 'None' else None,
            outro_video=args.outro_video if args.outro_video != 'None' else None,
            background_music=args.background_music if args.background_music != 'None' else None,
            pexels_keywords=json.loads(args.pexels_keywords),
            pexels_api_key=args.pexels_api_key,
            youtube_upload=youtube_upload_settings
        )
        
        # Validate input files only if not using Pexels
        logging.info("Validating input files...")
        if not profile.use_pexels:
            for img in args.scene_images:
                if not os.path.exists(img):
                    raise FileNotFoundError(f"Scene image not found: {img}")
        
        if not os.path.exists(args.final_audio_file):
            raise FileNotFoundError(f"Audio file not found: {args.final_audio_file}")
        
        temp_dir = args.temp_dir
        output_dir = f'{temp_dir}/Output/{profile.profile_name}'
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(f'{output_dir}/Videos', exist_ok=True)
        
        # Choose between local or EC2 video generation
        if args.use_local_generation:
            logging.info("Starting local video generation...")
            video_file = create_local_video(args, profile, output_dir)
        else:
            logging.info("Starting EC2 video generation...")
            create_video(args, profile, temp_dir, output_dir)
            video_file = f'{output_dir}/Videos/FinalVideo.mp4'

        if args.send_to_local:
            logging.info("Preparing video for local transfer...")
            # The actual transfer will be handled by the local_manager.py script
        elif isinstance(youtube_upload_settings, dict) and youtube_upload_settings.get('enabled', False):
            logging.info("Starting YouTube upload process...")
            try:
                # Extract all YouTube upload settings
                thumbnail_path = f'{temp_dir}/thumbnail.png'
                title = args.title
                description = youtube_upload_settings.get('description', '')
                tags = youtube_upload_settings.get('tags', [])
                category = youtube_upload_settings.get('category', '22')
                privacy_status = youtube_upload_settings.get('privacy_status', 'private')
                
                # Get the next scheduled upload date
                next_upload_date_str = youtube_upload_settings.get('next_upload_date')
                if not next_upload_date_str:
                    logging.error("No upload date provided in YouTube settings")
                    raise ValueError("Missing next_upload_date in YouTube settings")

                video_id = upload_to_youtube(
                    video_file=video_file,
                    title=title,
                    description=description,
                    tags=tags,
                    category=category,
                    privacy_status=privacy_status,
                    credentials_info=youtube_credentials,
                    publish_at=next_upload_date_str,
                    thumbnail_path=thumbnail_path
                )

                if video_id:
                    logging.info(f"Video successfully scheduled for upload to YouTube. Video ID: {video_id}")
                    logging.info(f"Scheduled publish time: {next_upload_date_str}")
                else:
                    logging.error("Failed to upload video to YouTube - no video ID returned")
                    
            except Exception as e:
                logging.error(f"Failed to upload video to YouTube: {str(e)}", exc_info=True)
                raise

        # Calculate and print the total time taken
        shutil.rmtree(temp_dir)
        end_time = time.time()
        total_duration = end_time - start_time
        formatted_time = str(timedelta(seconds=int(total_duration)))
        logging.info(f"\nTotal time taken to create the video: {formatted_time}\n")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Generate video from text and image')
        parser.add_argument('--scene_images', nargs='+', required=True, help='Paths to the scene background images')
        parser.add_argument('--scene_durations', nargs='+', type=int, required=True, help='Durations of each scene in milliseconds')
        parser.add_argument('--final_audio_file', required=True, help='Path to the final audio file')
        parser.add_argument('--profile_name', required=True, help='Profile name')
        parser.add_argument('--add_subtitles', type=lambda x: x.lower() == 'true', required=True,
                            help='Whether to add subtitles')
        parser.add_argument('--subtitle_style', required=True, help='Subtitle style configuration as JSON string')
        parser.add_argument('--audio_viz_config', required=True, help='Audio visualization configuration as JSON string')
        parser.add_argument('--use_pexels', type=lambda x: x.lower() == 'true', required=True,
                            help='Whether to use Pexels videos')
        parser.add_argument('--intro_video', help='Path to intro video')
        parser.add_argument('--outro_video', help='Path to outro video')
        parser.add_argument('--background_music', help='Path to background music')
        parser.add_argument('--pexels_keywords', required=True, help='Keywords for Pexels video search as JSON string')
        parser.add_argument('--pexels_api_key', required=True, help='Pexels API key')
        parser.add_argument('--youtube_upload', required=True, help='YouTube upload settings as JSON string')
        parser.add_argument('--youtube_credentials', required=True, help='YouTube credentials as JSON string')
        parser.add_argument('--title', required=True, help='Video title')
        parser.add_argument('--send_to_local', type=lambda x: x.lower() == 'true', required=True,
                            help='Whether to send the video back to local machine instead of uploading to YouTube')
        parser.add_argument('--temp_dir', required=True, help='Temporary directory for file operations')
        parser.add_argument('--use_local_generation', type=lambda x: x.lower() == 'true', required=True,
                            help='Whether to use local video generation instead of EC2')

        args = parser.parse_args()
        main(args)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(2)  # Exit with error code 2

