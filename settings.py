from pymongo import MongoClient
from typing import Dict, Any
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
import certifi
import json

# General Settings

# OpenAI API key for GPT models
OAI_API_KEY = None

# OpenRouter API key for accessing various AI models
OR_API_KEY = None

# ElevenLabs API key for text-to-speech
VOICE_API_KEY = None

# Pexels API key for accessing stock images and videos
PEXELS_API_KEY = None

REPLICATE_API_KEY = None

# Reddit API credentials
REDDIT_CLIENT_ID = None
REDDIT_CLIENT_SECRET = None
REDDIT_USER_AGENT = "Reddit posts"

# AI Model Settings
# Default model for general use (options: 'openai/gpt-3.5-turbo', 'openai/gpt-4', 'anthropic/claude-2', etc.)
MODEL = None

# Fine-tuned model for specific tasks
FT_MODEL = None

# Voice model for ElevenLabs
VOICE_MODEL = None

# AWS EC2 Settings for Video Creation
INSTANCE_TYPE = None

# AWS Instance storage size
INSTANCE_STORAGE = None

# Choose to run the video generator with simultaneous instances for each profile
CONCURRENT_EC2_INSTANCES = False

# Amazon Machine Image ID (AMI ID) for the EC2 instance
AMI_ID = None

# Name of the key pair for SSH access to the EC2 instance
KEY_NAME = None

# Path to the private key file for SSH access
KEY_FILE = None

# Security group name for the EC2 instance
SECURITY_GROUP = None

# Name tag for the EC2 instance
BASE_INSTANCE_NAME = None

# AWS credentials
AWS_ACCESS_KEY = None
AWS_SECRET_KEY = None

# AWS region for the EC2 instance
REGION = None

# Video Generation Settings
USE_LOCAL_GENERATION = True  # Default to False for EC2 generation

# MongoDB connection
def get_user_settings(username: str) -> Dict[str, Any]:
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["everything"]

    # Find document where username exists
    document = collection.find_one({username: {"$exists": True}})
    
    if not document:
        raise ValueError(f"No user found with username: {username}")
    
    user_settings = document[username].get('user-settings')
    
    if not user_settings:
        raise ValueError(f"No settings found for user: {username}")
    
    return user_settings

def get_channel_settings(username, channel_name):
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["everything"]

    # Find document where username exists
    document = collection.find_one({username: {"$exists": True}})
    
    if not document:
        raise ValueError(f"No user found with username: {username}")
    
    channels = document[username].get('channels', {})
    channel_settings = channels.get(channel_name)
    
    if not channel_settings:
        raise ValueError(f"No settings found for channel: {channel_name}")
    
    return channel_settings

def initialize_settings(username: str):
    """Initialize all settings from MongoDB based on username"""
    print("Starting settings initialization...")
    settings = get_user_settings(username)
    
    global OAI_API_KEY, OAI_MODEL, FT_MODEL, OR_API_KEY, VOICE_API_KEY, VOICE_MODEL, USE_FINE_TUNE, OR_MODEL
    global REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    global REPLICATE_API_KEY, PEXELS_API_KEY
    global AWS_ACCESS_KEY, AWS_SECRET_KEY, AMI_ID, INSTANCE_TYPE, KEY_FILE, KEY_NAME, BASE_INSTANCE_NAME, REGION, SECURITY_GROUP
    
    # OpenAI settings
    OAI_API_KEY = settings['open-ai']['open-ai-api-key']
    OR_API_KEY = settings['open-router']['open-router-api-key']
    OR_MODEL = settings['open-router']['model']
    OAI_MODEL = settings['open-ai']['model']
    FT_MODEL = settings['open-ai']['ft-model']

    # ElevenLabs settings
    VOICE_API_KEY = settings['elevenlabs']['elab_api_key']
    VOICE_MODEL = settings['elevenlabs']['elab_voice_model']

    # Replicate settings
    REPLICATE_API_KEY = settings['replicate']['replicate-api-key']
    
    # Pexels settings
    PEXELS_API_KEY = settings['pexels']['pexels-api-key']
    
    # Reddit settings
    REDDIT_CLIENT_ID = settings['reddit']['reddit-client-id']
    REDDIT_CLIENT_SECRET = settings['reddit']['reddit-client-secret']
    
    # AWS settings
    AWS_ACCESS_KEY = settings['aws']['aws_access_key']
    AWS_SECRET_KEY = settings['aws']['aws_secret_key']
    AMI_ID = settings['aws']['ami_id']
    INSTANCE_TYPE = settings['aws']['instance_type']
    BASE_INSTANCE_NAME = settings['aws']['base_instance_name']
    REGION = 'us-east-1'

def initialize_channel_settings(username: str, channel_name: str):
    """Initialize channel settings from MongoDB based on username and channel name"""
    channel_settings = get_channel_settings(username, channel_name)
    
    # Update global variables with channel settings
    global YOUTUBE_UPLOAD_ENABLED, YOUTUBE_DESCRIPTION, YOUTUBE_TAGS, YOUTUBE_CATEGORY
    global YOUTUBE_PRIVACY_STATUS, NEXT_UPLOAD_DATE, STORY_PROFILE, USE_REDDIT, USE_FINE_TUNE
    global STORY_TITLE_FT_MODEL, VOICE_ID, NUM_SCENES, ORIGINAL_IMAGE, ADD_SUBTITLES
    global AUDIO_VIZ_CONFIG, SUBTITLE_STYLE, THUMBNAIL_FONT, THUMBNAIL_PRIMARY_COLOR
    global THUMBNAIL_SECONDARY_COLOR, THUMBNAIL_STROKE_COLOR, THUMBNAIL_STROKE_WIDTH
    global COLOR_METHOD, USE_PEXELS, PEXELS_KEYWORDS, INTRO_VIDEO, OUTRO_VIDEO, BACKGROUND_MUSIC
    
    # YouTube Upload Settings
    YOUTUBE_UPLOAD_ENABLED = channel_settings['youtube']['upload']['enabled']
    YOUTUBE_DESCRIPTION = channel_settings['youtube']['upload']['description']
    YOUTUBE_TAGS = channel_settings['youtube']['upload']['tags']
    YOUTUBE_CATEGORY = channel_settings['youtube']['upload']['category']
    YOUTUBE_PRIVACY_STATUS = channel_settings['youtube']['upload']['privacy_status']
    NEXT_UPLOAD_DATE = channel_settings['youtube']['upload']['next_upload_date']
    
    # Story Settings
    STORY_PROFILE = channel_settings['story']['profile']
    USE_REDDIT = channel_settings['story']['use_reddit']
    STORY_TITLE_FT_MODEL = channel_settings['story']['title_ft_model']
    NUM_SCENES = channel_settings['story']['num_scenes']
    USE_FINE_TUNE = channel_settings['story']['use_fine_tune']

    
    # Audio Settings
    VOICE_ID = channel_settings['audio']['voice_id']
    BACKGROUND_MUSIC = channel_settings['audio']['background_music']
    
    # Video Settings
    ORIGINAL_IMAGE = channel_settings['video']['original_image']
    INTRO_VIDEO = channel_settings['video']['intro_video']
    OUTRO_VIDEO = channel_settings['video']['outro_video']
    USE_PEXELS = channel_settings['video']['use_pexels']
    PEXELS_KEYWORDS = channel_settings['video']['pexels_keywords']
    
    # Subtitle Settings
    ADD_SUBTITLES = channel_settings['subtitles']['enabled']
    SUBTITLE_STYLE = channel_settings['subtitles']['style']
    
    # Audio Visualization Settings
    AUDIO_VIZ_CONFIG = channel_settings['audio_visualization']
    
    # Thumbnail Settings
    COLOR_METHOD = channel_settings['thumbnail']['color_method']
    THUMBNAIL_FONT = channel_settings['thumbnail']['font']
    THUMBNAIL_PRIMARY_COLOR = channel_settings['thumbnail']['primary_color']
    THUMBNAIL_SECONDARY_COLOR = channel_settings['thumbnail']['secondary_color']
    THUMBNAIL_STROKE_COLOR = channel_settings['thumbnail']['stroke_color']
    THUMBNAIL_STROKE_WIDTH = channel_settings['thumbnail']['stroke_width']

def update_next_upload_date(username: str, channel_name: str):
    """Update the next_upload_date by adding 1 day"""
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["everything"]
    
    # Get current next_upload_date
    global NEXT_UPLOAD_DATE
    current_date = datetime.fromisoformat(NEXT_UPLOAD_DATE)
    
    # Add 1 day
    new_date = current_date + timedelta(days=1)
    new_date_str = new_date.isoformat()
    
    # Update MongoDB
    collection.update_one(
        {username: {"$exists": True}},
        {"$set": {f"{username}.channels.{channel_name}.youtube.upload.next_upload_date": new_date_str}}
    )
    
    # Update global variable
    NEXT_UPLOAD_DATE = new_date_str

def get_channel_names(username: str) -> list:
    """Get a list of all channel names for a given username"""
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["everything"]
    
    # Find document where username exists
    document = collection.find_one({username: {"$exists": True}})
    
    if not document:
        return []
    
    channels = document[username].get('channels', {})
    return list(channels.keys())

def load_story_profiles():
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["video-types"]

    try:
        story_profiles = collection.find_one()
        if not story_profiles:
            print("Error: No story profiles found in the video-types collection.")
            return {}
        return story_profiles
    except Exception as e:
        print(f"Error: {e}")
        return {}

def copy_channels_between_users(from_username: str, to_username: str):
    """
    Copy all channels from one user to another user in the MongoDB database.
    
    Args:
        from_username (str): Username to copy channels from
        to_username (str): Username to copy channels to
    """
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["everything"]

    # Get source user document
    source_doc = collection.find_one({from_username: {"$exists": True}})
    if not source_doc:
        raise ValueError(f"Source user {from_username} not found")
    
    # Get target user document
    target_doc = collection.find_one({to_username: {"$exists": True}})
    if not target_doc:
        raise ValueError(f"Target user {to_username} not found")
    
    # Get channels from source user
    source_channels = source_doc[from_username].get('channels', {})
    
    # Update target user's channels
    collection.update_one(
        {to_username: {"$exists": True}},
        {"$set": {f"{to_username}.channels": source_channels}}
    )

def duplicate_channel(username: str, source_channel: str, new_channel_name: str):
    """
    Duplicate a specific channel for a user with a new name.
    
    Args:
        username (str): Username whose channel to duplicate
        source_channel (str): Name of the channel to duplicate
        new_channel_name (str): Name for the new duplicate channel
    """
    client = MongoClient(
        'mongodb+srv://TheRealceCream:xijj69DyfnQOXD9d@cluster0.xvcs7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        tlsCAFile=certifi.where()
    )
    db = client['YouTube-Dashboard']
    collection = db["everything"]

    # Get user document
    document = collection.find_one({username: {"$exists": True}})
    if not document:
        raise ValueError(f"User {username} not found")
    
    # Get source channel settings
    channels = document[username].get('channels', {})
    if source_channel not in channels:
        raise ValueError(f"Source channel {source_channel} not found")
    
    if new_channel_name in channels:
        raise ValueError(f"Channel name {new_channel_name} already exists")
    
    # Create a deep copy of the channel settings
    new_channel_settings = json.loads(json.dumps(channels[source_channel]))
    
    # Update MongoDB with the new channel
    collection.update_one(
        {username: {"$exists": True}},
        {"$set": {f"{username}.channels.{new_channel_name}": new_channel_settings}}
    )

USE_SINGLE_IMAGE = False  # or False to use multiple images

if __name__ == "__main__":
    # print(get_channel_names('229202'))
    # copy_channels_between_users('229202', '327765')
    duplicate_channel('327765', 'Relationship Haven', 'Horror Channel')
