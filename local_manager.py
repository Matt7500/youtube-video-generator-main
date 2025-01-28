import boto3
import os
import time
import paramiko
import select
import uuid
import settings
import json
from datetime import datetime, timedelta
from threading import Lock
import shutil
from botocore.exceptions import ClientError
import botocore.exceptions
from botocore.config import Config

active_instances = {}
instance_lock = Lock()


def get_youtube_credentials(profile_name):
    creds_file = f'credentials/youtube_credentials_{profile_name}.json'
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"YouTube credentials not found for profile '{profile_name}'. "
                                f"Please run the authentication script for this profile first.")

    with open(creds_file, 'r') as f:
        return json.load(f)

def run_command_with_realtime_output(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command)

    # Set up channel for streaming
    channel = stdout.channel
    channel.settimeout(0.0)

    # Stream output
    while not channel.closed or channel.recv_ready() or channel.recv_stderr_ready():
        readq, _, _ = select.select([channel], [], [], 0.1)
        for c in readq:
            if c.recv_ready():
                print(c.recv(1024).decode('utf-8', 'ignore'), end='', flush=True)
            if c.recv_stderr_ready():
                print(c.recv_stderr(1024).decode('utf-8', 'ignore'), end='', flush=True)

        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break

    exit_status = channel.recv_exit_status()
    return exit_status


def create_security_group_if_needed(ec2):
    """Create or get security group with necessary permissions."""
    group_name = f"{settings.BASE_INSTANCE_NAME}-sg"
    
    try:
        # Check if security group already exists
        response = ec2.describe_security_groups(GroupNames=[group_name])
        return response['SecurityGroups'][0]['GroupId']
    except ClientError as e:
        if e.response['Error']['Code'] != 'InvalidGroup.NotFound':
            raise

        print(f"Creating new security group: {group_name}")
        
        # Create security group
        sg = ec2.create_security_group(
            GroupName=group_name,
            Description='Security group for video generation instances'
        )
        
        # Add inbound rules
        ec2.authorize_security_group_ingress(
            GroupId=sg['GroupId'],
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,  # SSH
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        return sg['GroupId']

def create_key_pair_if_needed(ec2, username):
    """Create or get key pair for EC2 instance."""
    key_name = f"{username}-VideoGenerator-key"
    key_file = f"credentials/{key_name}.pem"
    
    try:
        # Check if key pair exists in AWS
        ec2.describe_key_pairs(KeyNames=[key_name])
        
        # Check if we have the private key file
        if not os.path.exists(key_file):
            raise FileNotFoundError(
                f"Key pair exists in AWS but private key file not found: {key_file}"
            )
        
        return key_name, key_file
    
    except ClientError as e:
        if e.response['Error']['Code'] != 'InvalidKeyPair.NotFound':
            raise
            
        print(f"Creating new key pair: {key_name}")
        
        # Create directory if it doesn't exist
        os.makedirs('credentials', exist_ok=True)
        
        # Create new key pair
        key_pair = ec2.create_key_pair(KeyName=key_name)
        
        # Save private key
        with open(key_file, 'w') as f:
            f.write(key_pair['KeyMaterial'])
        
        # Set correct permissions for key file
        os.chmod(key_file, 0o600)
        
        return key_name, key_file

def get_or_create_instance(ec2, username, profile_name, instance_type, ami_id):
    """Get existing instance or create new one with proper security group and key pair."""
    instance_name = f"{username}-VideoGenerator"
    max_retries = 3
    base_delay = 5  # seconds

    # Configure the client with longer timeouts - Fixed Config import
    config = Config(
        connect_timeout=120,  # 2 minutes
        read_timeout=120,
        retries={'max_attempts': 3}
    )
    ec2 = boto3.client('ec2',
                      aws_access_key_id=settings.AWS_ACCESS_KEY,
                      aws_secret_access_key=settings.AWS_SECRET_KEY,
                      region_name=settings.REGION,
                      config=config)

    for attempt in range(max_retries):
        try:
            # Ensure we have security group and key pair
            security_group_id = create_security_group_if_needed(ec2)
            key_name, key_file = create_key_pair_if_needed(ec2, username)
            
            # Update settings with the key file path
            settings.KEY_FILE = key_file
            settings.KEY_NAME = key_name
            settings.SECURITY_GROUP = security_group_id

            # Check for existing instance
            instances = ec2.describe_instances(Filters=[
                {'Name': 'tag:Name', 'Values': [instance_name]},
                {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}
            ])['Reservations']

            if instances:
                instance_id = instances[0]['Instances'][0]['InstanceId']
                instance_state = instances[0]['Instances'][0]['State']['Name']

                if instance_state == 'stopped':
                    print(f"Starting existing instance {instance_id} for profile {profile_name}")
                    try:
                        ec2.start_instances(InstanceIds=[instance_id])
                        waiter = ec2.get_waiter('instance_running')
                        waiter.wait(
                            InstanceIds=[instance_id],
                            WaiterConfig={'Delay': 5, 'MaxAttempts': 40}
                        )
                    except Exception as e:
                        print(f"Error starting instance: {str(e)}")
                        if attempt < max_retries - 1:
                            continue
                        raise
                else:
                    print(f"Reusing running instance {instance_id} for profile {profile_name}")

                return instance_id, False

            # Create new instance if none exists
            print(f"Creating new EC2 instance for profile {profile_name}")
            instance = ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                KeyName=key_name,
                SecurityGroupIds=[security_group_id],
                MinCount=1,
                MaxCount=1,
                BlockDeviceMappings=[
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': 500,
                            'VolumeType': 'gp2'
                        }
                    }
                ],
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [{'Key': 'Name', 'Value': instance_name}]
                    }
                ]
            )['Instances'][0]

            instance_id = instance['InstanceId']
            waiter = ec2.get_waiter('instance_running')
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={'Delay': 5, 'MaxAttempts': 40}
            )

            return instance_id, True

        except (botocore.exceptions.ConnectTimeoutError,
                botocore.exceptions.EndpointConnectionError,
                botocore.exceptions.ClientError) as e:
            if attempt < max_retries - 1:
                delay = (base_delay * (2 ** attempt))  # exponential backoff
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Unable to connect to AWS.")
                raise

    raise Exception("Failed to create or get instance after all retries")

def install_dependencies(ssh, is_new_instance):
    if is_new_instance:
        print("Setting up new EC2 instance...")
        commands = [
            "sudo apt update",
            "sudo apt upgrade -y",
            "sudo apt install -y python3-pip python3-venv ffmpeg",
            "python3 -m venv /home/ubuntu/video_env",
            "source /home/ubuntu/video_env/bin/activate",
            "/home/ubuntu/video_env/bin/pip install --upgrade pip setuptools wheel",
            # Install MoviePy and other required packages
            "/home/ubuntu/video_env/bin/pip install moviepy ffmpeg-python",
            # Then install the rest of your existing dependencies
            "/home/ubuntu/video_env/bin/pip install " + " ".join([
                "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                # ... rest of your existing packages ...
            ])
        ]

        for cmd in commands:
            run_command_with_realtime_output(ssh, cmd)
    else:
        # For existing instances, just ensure MoviePy is installed
        commands = [
            "source /home/ubuntu/video_env/bin/activate",
            "/home/ubuntu/video_env/bin/pip install --upgrade moviepy ffmpeg-python"
        ]
        for cmd in commands:
            run_command_with_realtime_output(ssh, cmd)

def create_video_on_instance(username, channel_name, scene_images, scene_durations, final_audio_file, thumbnail_path, chosen_title):
    send_to_local = False

    # Get YouTube upload settings
    youtube_upload_settings = {
        'enabled': settings.YOUTUBE_UPLOAD_ENABLED,
        'description': settings.YOUTUBE_DESCRIPTION,  # Add this to settings
        'tags': settings.YOUTUBE_TAGS,  # Add this to settings
        'category': settings.YOUTUBE_CATEGORY,  # Add this to settings (default: '22' for People & Blogs)
        'privacy_status': settings.YOUTUBE_PRIVACY_STATUS,  # Add this to settings
        'next_upload_date': settings.NEXT_UPLOAD_DATE  # Convert datetime to string
    }

    # Create EC2 client
    ec2 = boto3.client('ec2', 
                      aws_access_key_id=settings.AWS_ACCESS_KEY, 
                      aws_secret_access_key=settings.AWS_SECRET_KEY,
                      region_name=settings.REGION)

    # Get or create EC2 instance
    instance_id, is_new_instance = get_or_create_instance(ec2, username, channel_name, settings.INSTANCE_TYPE, settings.AMI_ID)
    print(f"Using EC2 instance {instance_id} for channel {channel_name}")

    # Get the public IP address of the instance
    instance_info = ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]
    public_ip = instance_info['PublicIpAddress']

    # Connect to the instance using SSH
    key = paramiko.RSAKey.from_private_key_file(settings.KEY_FILE)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Add a retry mechanism for SSH connection
    max_retries = 5
    for attempt in range(max_retries):
        try:
            ssh.connect(hostname=public_ip, username='ubuntu', pkey=key, timeout=60)
            print(f"Successfully connected to EC2 instance {instance_id}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed for instance {instance_id}. Retrying in 30 seconds...")
                time.sleep(30)
            else:
                raise Exception(f"Failed to connect to instance {instance_id} after {max_retries} attempts: {str(e)}")

    try:
        # Install dependencies (only full setup for new instances)
        print(f"{'Installing' if is_new_instance else 'Updating'} dependencies on instance {instance_id}")
        install_dependencies(ssh, is_new_instance)

        # Create a temporary directory on the EC2 instance
        temp_dir = f'/tmp/video_gen_{uuid.uuid4().hex}'
        ssh.exec_command(f'mkdir -p {temp_dir}')

        # Transfer files to EC2 instance
        sftp = ssh.open_sftp()
        print(f"Transferring files to EC2 instance {instance_id}")

        # Transfer scene images only if not using Pexels
        remote_scene_images = []
        if not settings.USE_PEXELS:
            for i, image_path in enumerate(scene_images, 1):
                remote_path = f'{temp_dir}/scene_{i}_background.png'
                sftp.put(image_path, remote_path)
                remote_scene_images.append(remote_path)
                print(f'Scene {i} background image transferred.')
        else:
            # When using Pexels, we'll just pass empty paths
            remote_scene_images = [f'{temp_dir}/scene_{i}_background.png' for i in range(1, len(scene_images) + 1)]

        # Transfer final audio file
        remote_final_audio = f'{temp_dir}/Final.mp3'
        sftp.put(final_audio_file, remote_final_audio)
        print('Final audio file transferred.')

        # Transfer other required files
        sftp.put('vid_gen.py', f'{temp_dir}/vid_gen.py')
        print('Video Generator transferred.')
        remote_thumbnail = f'{temp_dir}/thumbnail.png'
        sftp.put(thumbnail_path, remote_thumbnail)
        print('Thumbnail image transferred.')

        # Update paths for channel-specific files
        remote_intro = "None"
        remote_outro = "None"
        remote_bg_music = "None"

        if settings.INTRO_VIDEO and settings.INTRO_VIDEO.lower() != "null":
            remote_intro = f'{temp_dir}/{os.path.basename(settings.INTRO_VIDEO)}'
            sftp.put(settings.INTRO_VIDEO, remote_intro)
            print('Intro video transferred.')
            
        if settings.OUTRO_VIDEO and settings.OUTRO_VIDEO.lower() != "null":
            remote_outro = f'{temp_dir}/{os.path.basename(settings.OUTRO_VIDEO)}'
            sftp.put(settings.OUTRO_VIDEO, remote_outro)
            print('Outro video transferred.')
            
        if settings.BACKGROUND_MUSIC and settings.BACKGROUND_MUSIC.lower() != "null":
            remote_bg_music = f'{temp_dir}/{os.path.basename(settings.BACKGROUND_MUSIC)}'
            sftp.put(settings.BACKGROUND_MUSIC, remote_bg_music)
            print('Background music transferred.')

        # Prepare YouTube upload settings
        if settings.YOUTUBE_UPLOAD_ENABLED:
            youtube_credentials = get_youtube_credentials(channel_name)
        else:
            youtube_credentials = {}
            youtube_upload_settings['enabled'] = False

        # Run the video creation script with updated parameters
        print(f"Running video creation script on EC2 instance {instance_id}")
        command = f'/home/ubuntu/video_env/bin/python3 {temp_dir}/vid_gen.py ' \
                  f'--profile_name "{channel_name}" ' \
                  f'--scene_images {" ".join(remote_scene_images)} ' \
                  f'--scene_durations {" ".join(map(str, scene_durations))} ' \
                  f'--final_audio_file "{remote_final_audio}" ' \
                  f'--add_subtitles "{str(settings.ADD_SUBTITLES).lower()}" ' \
                  f'--audio_viz_config \'{json.dumps(settings.AUDIO_VIZ_CONFIG)}\' ' \
                  f'--subtitle_style \'{json.dumps(settings.SUBTITLE_STYLE)}\' ' \
                  f'--use_pexels "{str(settings.USE_PEXELS).lower()}" ' \
                  f'--intro_video "{remote_intro}" ' \
                  f'--outro_video "{remote_outro}" ' \
                  f'--background_music "{remote_bg_music}" ' \
                  f'--pexels_keywords \'{json.dumps(settings.PEXELS_KEYWORDS)}\' ' \
                  f'--pexels_api_key "{settings.PEXELS_API_KEY}" ' \
                  f'--youtube_upload \'{json.dumps(youtube_upload_settings)}\' ' \
                  f'--youtube_credentials \'{json.dumps(youtube_credentials)}\' ' \
                  f'--title "{chosen_title}" ' \
                  f'--send_to_local "{str(send_to_local).lower()}" ' \
                  f'--temp_dir "{temp_dir}"'

        exit_status = run_command_with_realtime_output(ssh, command)
        
        if exit_status != 0:
            print(f"Error: vid_gen.py exited with status {exit_status} for channel {channel_name}")
        else:
            print(f"vid_gen.py completed successfully for channel {channel_name}")

        if send_to_local:
            # Transfer the output video back to local machine
            print(f"Transferring output video back from instance {instance_id}")
            os.makedirs(f'Output/{channel_name}/Videos', exist_ok=True)
            
            # List all files in temp directory to debug
            stdin, stdout, stderr = ssh.exec_command(f'find {temp_dir} -name "FinalVideo.mp4"')
            files = stdout.read().decode().strip()
            print(f"Found video files: {files}")
            
            if files:
                remote_video_path = files.split('\n')[0]  # Take the first match if multiple exist
                local_video_path = f'Output/{channel_name}/Videos/FinalVideo.mp4'
                print(f"Attempting to retrieve file from: {remote_video_path}")
                try:
                    sftp.get(remote_video_path, local_video_path)
                    print(f"Video successfully transferred to: {local_video_path}")
                except IOError as e:
                    print(f"Error: Unable to retrieve video file. {str(e)}")
            else:
                print("No video file found on remote instance")
                print("Listing contents of the temp directory:")
                stdin, stdout, stderr = ssh.exec_command(f'ls -R {temp_dir}')
                print(stdout.read().decode())
        elif settings.YOUTUBE_UPLOAD_ENABLED:
            # Update the next upload date in MongoDB
            settings.update_next_upload_date(username, channel_name)

    finally:
        # Clean up the temporary directory
        ssh.exec_command(f'rm -rf {temp_dir}')
        
        # Close SSH connection
        ssh.close()
        
        # Stop the EC2 instance
        print(f"Stopping EC2 instance {instance_id} for channel {channel_name}")
        ec2.stop_instances(InstanceIds=[instance_id])
        
        # Remove this instance from active instances
        with instance_lock:
            if instance_id in active_instances:
                active_instances[instance_id].remove(channel_name)
                if not active_instances[instance_id]:
                    del active_instances[instance_id]

        print(f"Video creation process completed and instance {instance_id} stopped for channel {channel_name}.")


def main(profile_name, scene_images, scene_audio_files, scene_durations, final_audio_file, thumbnail_path, chosen_title):
    global active_instances

    # Create EC2 client
    ec2 = boto3.client('ec2', aws_access_key_id=settings.AWS_ACCESS_KEY, aws_secret_access_key=settings.AWS_SECRET_KEY,
                       region_name=settings.REGION)

    # Get or create EC2 instance
    instance_id, is_new_instance = get_or_create_instance(ec2, profile_name, settings.INSTANCE_TYPE, settings.AMI_ID,
                                                          settings.KEY_NAME, settings.SECURITY_GROUP)
    
    # Add this instance and profile to active_instances
    with instance_lock:
        if instance_id not in active_instances:
            active_instances[instance_id] = set()
        active_instances[instance_id].add(profile_name)

    create_video_on_instance(profile_name, scene_images, scene_durations, final_audio_file, thumbnail_path, chosen_title)

def connect_to_instance(instance_id):
    """Connect to an EC2 instance and return SSH client."""
    # Create EC2 client
    ec2 = boto3.client('ec2', 
                      aws_access_key_id=settings.AWS_ACCESS_KEY, 
                      aws_secret_access_key=settings.AWS_SECRET_KEY,
                      region_name=settings.REGION)

    # Get the public IP address of the instance
    instance_info = ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]
    public_ip = instance_info['PublicIpAddress']

    # Connect to the instance using SSH
    key = paramiko.RSAKey.from_private_key_file(settings.KEY_FILE)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Add a retry mechanism for SSH connection
    max_retries = 5
    for attempt in range(max_retries):
        try:
            ssh.connect(hostname=public_ip, username='ubuntu', pkey=key, timeout=60)
            print(f"Successfully connected to EC2 instance {instance_id}")
            return ssh
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed for instance {instance_id}. Retrying in 30 seconds...")
                time.sleep(30)
            else:
                raise Exception(f"Failed to connect to instance {instance_id} after {max_retries} attempts: {str(e)}")

def check_instance_packages():
    """Connect to instance and run pip list."""
    username = "check_packages"
    profile_name = "test"
    
    # Create EC2 client
    config = Config(
        connect_timeout=120,  # 2 minutes
        read_timeout=120,
        retries={'max_attempts': 3}
    )
    ec2 = boto3.client('ec2',
                      aws_access_key_id=settings.AWS_ACCESS_KEY,
                      aws_secret_access_key=settings.AWS_SECRET_KEY,
                      region_name=settings.REGION,
                      config=config)
    
    try:
        # Get or create instance using the updated function
        instance_id, is_new_instance = get_or_create_instance(
            ec2,
            username,
            profile_name,
            settings.INSTANCE_TYPE,
            settings.AMI_ID
        )
        
        # Connect to instance using SSH
        ssh = connect_to_instance(instance_id)
        
        try:
            # Install dependencies using the existing function
            install_dependencies(ssh, is_new_instance)
            
            # Additional font installations
            print("\nInstalling additional fonts:")
            commands = [
                "sudo apt install -y fonts-freefont-ttf",
                "sudo apt install -y fonts-noto",
                "fc-list"
            ]
            
            for cmd in commands:
                run_command_with_realtime_output(ssh, cmd)

        finally:
            ssh.close()
            
    except Exception as e:
        print(f"Error during instance check: {str(e)}")
        raise
    finally:
        # Stop the instance when done
        if 'instance_id' in locals():
            print(f"Stopping EC2 instance {instance_id}")
            ec2.stop_instances(InstanceIds=[instance_id])

def create_local_video(username, channel_name, scene_images, scene_durations, final_audio_path, thumbnail_path, title):
    """Create video locally using vid_gen.py"""
    print(f"Creating local video for channel: {channel_name}")
    
    # Prepare YouTube upload settings
    youtube_upload_settings = {
        'enabled': settings.YOUTUBE_UPLOAD_ENABLED,
        'description': settings.YOUTUBE_DESCRIPTION,
        'tags': settings.YOUTUBE_TAGS,
        'category': settings.YOUTUBE_CATEGORY,
        'privacy_status': settings.YOUTUBE_PRIVACY_STATUS,
        'next_upload_date': settings.NEXT_UPLOAD_DATE
    }

    # Get YouTube credentials if needed
    if settings.YOUTUBE_UPLOAD_ENABLED:
        youtube_credentials = get_youtube_credentials(channel_name)
    else:
        youtube_credentials = {}

    # Create a temporary directory with unique UUID
    temp_dir = f'/tmp/video_gen_{uuid.uuid4().hex}'
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Copy files to temp directory
        sftp = None
        print(f"Setting up files in temporary directory")

        # Copy scene images
        remote_scene_images = []
        if not settings.USE_PEXELS:
            for i, image_path in enumerate(scene_images, 1):
                remote_path = f'{temp_dir}/scene_{i}_background.png'
                shutil.copy2(image_path, remote_path)
                remote_scene_images.append(remote_path)
                print(f'Scene {i} background image copied.')
        else:
            remote_scene_images = [f'{temp_dir}/scene_{i}_background.png' for i in range(1, len(scene_images) + 1)]

        # Copy final audio file
        remote_final_audio = f'{temp_dir}/Final.mp3'
        shutil.copy2(final_audio_path, remote_final_audio)
        print('Final audio file copied.')

        # Copy thumbnail
        remote_thumbnail = f'{temp_dir}/thumbnail.png'
        shutil.copy2(thumbnail_path, remote_thumbnail)
        print('Thumbnail image copied.')

        # Copy other required files
        remote_intro = "None"
        remote_outro = "None"
        remote_bg_music = "None"

        if settings.INTRO_VIDEO and settings.INTRO_VIDEO.lower() != "null":
            remote_intro = f'{temp_dir}/{os.path.basename(settings.INTRO_VIDEO)}'
            shutil.copy2(settings.INTRO_VIDEO, remote_intro)
            print('Intro video copied.')
            
        if settings.OUTRO_VIDEO and settings.OUTRO_VIDEO.lower() != "null":
            remote_outro = f'{temp_dir}/{os.path.basename(settings.OUTRO_VIDEO)}'
            shutil.copy2(settings.OUTRO_VIDEO, remote_outro)
            print('Outro video copied.')
            
        if settings.BACKGROUND_MUSIC and settings.BACKGROUND_MUSIC.lower() != "null":
            remote_bg_music = f'{temp_dir}/{os.path.basename(settings.BACKGROUND_MUSIC)}'
            shutil.copy2(settings.BACKGROUND_MUSIC, remote_bg_music)
            print('Background music copied.')

        # Prepare arguments for vid_gen.py
        args = type('Args', (), {
            'profile_name': channel_name,
            'scene_images': remote_scene_images,
            'scene_durations': scene_durations,
            'final_audio_file': remote_final_audio,
            'add_subtitles': str(settings.ADD_SUBTITLES).lower(),
            'audio_viz_config': json.dumps(settings.AUDIO_VIZ_CONFIG),
            'subtitle_style': json.dumps(settings.SUBTITLE_STYLE),
            'use_pexels': str(settings.USE_PEXELS).lower(),
            'intro_video': remote_intro,
            'outro_video': remote_outro,
            'background_music': remote_bg_music,
            'pexels_keywords': json.dumps(settings.PEXELS_KEYWORDS),
            'pexels_api_key': settings.PEXELS_API_KEY,
            'youtube_upload': json.dumps(youtube_upload_settings),
            'youtube_credentials': json.dumps(youtube_credentials),
            'title': title,
            'send_to_local': 'true',
            'temp_dir': temp_dir,
            'use_local_generation': 'true'
        })()

        # Import vid_gen and create video
        import vid_gen
        profile = vid_gen.VideoProfile(
            add_subtitles=settings.ADD_SUBTITLES,
            profile_name=channel_name,
            audio_viz_config=settings.AUDIO_VIZ_CONFIG,
            subtitle_style=settings.SUBTITLE_STYLE,
            use_pexels=settings.USE_PEXELS,
            intro_video=remote_intro if remote_intro != "None" else None,
            outro_video=remote_outro if remote_outro != "None" else None,
            background_music=remote_bg_music if remote_bg_music != "None" else None,
            pexels_keywords=settings.PEXELS_KEYWORDS,
            pexels_api_key=settings.PEXELS_API_KEY,
            youtube_upload=youtube_upload_settings
        )

        video_file = vid_gen.create_local_video(args, profile, temp_dir)
        
        if settings.YOUTUBE_UPLOAD_ENABLED:
            # Update the next upload date in MongoDB
            settings.update_next_upload_date(username, channel_name)

        # Copy the final video to the output directory
        output_dir = f'Output/{channel_name}/Videos'
        os.makedirs(output_dir, exist_ok=True)
        final_video_path = os.path.join(output_dir, 'FinalVideo.mp4')
        shutil.copy2(video_file, final_video_path)

        print(f"Local video creation completed for channel {channel_name}")
        return final_video_path

    finally:
        # Clean up the temporary directory
        print(f"Cleaning up temporary directory")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    settings.initialize_settings('229202')
    check_instance_packages()

