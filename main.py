import story_writer
import local_operations
import local_manager
import settings
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from settings import initialize_settings, update_next_upload_date, get_channel_names, initialize_channel_settings  # Import the functions
import time  # Add this import at the top with other imports


def update_from_github():
    print("Checking for updates from private GitHub repository...")
    try:
        repo_url = 'https://github.com/Matt7500/youtube-video-generator'
        
        # Fetch the latest changes without merging
        subprocess.run(['git', 'fetch', repo_url], 
                       check=True, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE)
        
        # Check if there are any differences between local and remote
        result = subprocess.run(['git', 'diff', 'HEAD', 'FETCH_HEAD', '--name-only'],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        
        if result.stdout.strip():
            print("Updates found. Downloading new files...")
            # Pull the changes
            update_result = subprocess.run(['git', 'pull', repo_url],
                                           check=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           text=True)
            print("Update successful:")
            print(update_result.stdout)
        else:
            print("No updates found. Continuing with current files.")
    except subprocess.CalledProcessError as e:
        print(f"Error during GitHub update check: {e}")
        print(f"Error output: {e.stderr}")


def process_channel(username, channel_name):
    print(f"\n=== Starting process for channel: {channel_name} ===")

    # Delete the local channel folder
    local_channel_folder = f'Output/{channel_name}'
    if os.path.exists(local_channel_folder):
        print(f"Deleting contents of the local channel folder for {channel_name}")
        shutil.rmtree(local_channel_folder)
        print(f"Successfully deleted local channel folder contents for {channel_name}")

    try:
        print("Step 1: Calling story_writer.main...")
        result = story_writer.main(username, channel_name)
        
        story, scenes, story_idea = result
        print(f"Successfully unpacked story ({len(story)} chars), scenes ({len(scenes)} scenes), and story idea")

        # print("Stopping at this point")
        # sys.exit(0)
        
        print("\nStep 2: Starting local operations processing...")
        result = local_operations.process_local(username, channel_name, story, scenes, story_idea)
        
        if result is not None and len(result) == 6:
            print("\nStep 3: Unpacking local operations result...")
            scene_images, scene_audio_files, scene_durations, final_audio_path, thumbnail_path, title = result
            
            print(f"- Scene images: {len(scene_images) if scene_images else 'None'}")
            print(f"- Audio files: {len(scene_audio_files) if scene_audio_files else 'None'}")
            print(f"- Scene durations: {len(scene_durations) if scene_durations else 'None'}")
            print(f"- Final audio: {'Present' if final_audio_path else 'Missing'}")
            print(f"- Thumbnail: {'Present' if thumbnail_path else 'Missing'}")
            print(f"- Title: {title if title else 'Missing'}")
            
            if scene_images and scene_audio_files and scene_durations and final_audio_path and thumbnail_path and title:
                print("\nStep 4: Creating output directory...")
                os.makedirs(f'Output/{channel_name}', exist_ok=True)
                
                if settings.USE_LOCAL_GENERATION:
                    print("Using local video generation...")
                    local_manager.create_local_video(username, channel_name, scene_images, scene_durations, final_audio_path, thumbnail_path, title)
                    print(f"Channel {channel_name} processed successfully.")
                else:
                    print("Using EC2 video generation...")
                    local_manager.create_video_on_instance(username, channel_name, scene_images, scene_durations, final_audio_path, thumbnail_path, title)
                    print(f"Channel {channel_name} processed successfully.")
                
                return channel_name, scene_images, scene_audio_files, scene_durations, final_audio_path, thumbnail_path, title
            else:
                print("\nError: One or more required components are missing")
        else:
            print(f"\nError: Invalid result format from local_operations. Expected 6 items, got: {len(result) if result else 'None'}")
        
        print(f"\nAn error occurred during local operations for channel {channel_name}.")
        return None
    
    except Exception as e:
        print(f"\n!!! Exception in process_channel for {channel_name} !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def generate_video_on_ec2(username, channel_name, profile_data):
    if profile_data is None:
        print("Error: Invalid profile data")
        return
    
    profile_name, scene_images, scene_audio_files, scene_durations, final_audio_file, thumbnail_path, chosen_title = profile_data
    print(f"Generating video for profile: {profile_name}")
    local_manager.create_video_on_instance(username, channel_name, scene_images, scene_durations, final_audio_file, thumbnail_path, chosen_title)
    print(f"Video generated successfully for profile: {profile_name}")


def main(username, num_videos_per_channel):
    start_time = time.time()  # Start timing
    print("Starting main function...")
    # Initialize all settings once at the start
    initialize_settings(username)
    print("Settings initialized")
    
    channel_names = get_channel_names(username)
    print(f"Found channels: {channel_names}")
    
    # Initialize channel settings once for each channel
    for channel in channel_names:
        print(f"Initializing settings for channel: {channel}")
        initialize_channel_settings(username, channel)
    
    # Update files from GitHub before proceeding
    update_from_github()
    
    # Process channels sequentially
    processed_channels = []
    for i, channel in enumerate(channel_names):
        if i == 1:
            break
        for _ in range(num_videos_per_channel):
            result = process_channel(username, channel)
            if result:
                if not settings.USE_LOCAL_GENERATION:
                    generate_video_on_ec2(username, channel, result)
                processed_channels.append(result)

    print("All selected videos have been generated successfully.")
    
    # Calculate and print total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    main('229202', 1)
