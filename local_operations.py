import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
from openai import OpenAI
import shutil
from pydub import AudioSegment
from natsort import natsorted
from tqdm import tqdm
import settings
import time
from datetime import datetime
import os
from io import BytesIO
import replicate
from replicate.exceptions import ReplicateError
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import httpx

# Initialize clients - move this into a function
oai_client = None
or_client = None
replicate_client = None
elevenlabs_client = None

def initialize_clients(username):
    try:
        print(f"Initializing settings for user: {username}")
        settings.initialize_settings(username)
        
        # Add USE_SINGLE_IMAGE setting initialization
        settings.USE_SINGLE_IMAGE = getattr(settings, 'USE_SINGLE_IMAGE', False)
        
        print("Initializing API clients...")
        global oai_client, or_client, replicate_client, elevenlabs_client
        
        print("Setting up OpenAI client...")
        oai_client = OpenAI(api_key=settings.OAI_API_KEY)

        print("Setting up OpenRouter client...")
        or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=settings.OR_API_KEY
        )
        
        print("Setting up Replicate client...")
        replicate_client = replicate.Client(api_token=settings.REPLICATE_API_KEY)
        print(f"Replicate client initialized with API key: {settings.REPLICATE_API_KEY[:8]}...")
        
        print("Setting up ElevenLabs client...")
        elevenlabs_client = ElevenLabs(api_key=settings.VOICE_API_KEY)
        
        print("All API clients initialized successfully.")
        
    except Exception as e:
        print(f"Error initializing clients: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

def summarize_text(text, story_profile, is_thumbnail=False, max_retries=3, initial_delay=1):
    """Summarize text with retry logic for API failures"""
    for attempt in range(max_retries):
        try:
            # First summary attempt using OpenRouter
            message = or_client.chat.completions.create(
                model=settings.OR_MODEL,
                max_tokens=4000,
                temperature=0.5,
                messages=[
                    {"role": "system",
                     "content": f"Write a detailed summary of the given story. You must only respond with the summary, do not write any comments."},
                    {"role": "user", "content": text}
                ]
            )
            
            if story_profile == 'Horror':
                if is_thumbnail:
                    # Second summary attempt for thumbnails using OpenAI
                    try:
                        message2 = oai_client.chat.completions.create(
                            model='chatgpt-4o-latest',
                            max_tokens=4000,
                            temperature=0.7,
                            messages=[
                                {"role": "system",
                                "content": f"""Write a very short description of a scene that shows what the given story summary is about.
                                Include the people in the story or the monster/creature in the story as the subject of the scene.
                                It must describe the scene as it already is. Only write what is in the plot of the story and is important to the plot of the story.
                                Do not mention any children or people in the description, only describe the setting.
                                The description must not be nsfw or disturbing.
                                If there is a creature or entity in crucial to the story then describe that in a location in the story.
                                If there is a building or a house/cabin the story takes place in then describe that in it's location in the story.
                                Do not describe any actions in the scene, only describe what is visually there.
                                The creature/entity or building if exists must be the focus of the description.
                                Do not include any text in the description.
                                Do not mention any weapons in the description.
                                Write this in 20 words or less.
        
                                ## Examples:
                                a person sitting on the edge of a bed in a dark concrete room
                                a young girl standing in a playground at night
                                a man standing outside a house at night
                                a scary person in a diner at night
                                a large sea creature in the dark ocean
                                a forest ranger standing in front of a large creature at night
                                an abandoned mansion at night
                                a cop car in a cornfield at night
                                a lighthouse on a cliff overlooking the ocean at night
                                an old graveyard in a forest at night
                                a wooden fire lookout tower at night
                                a cabin in the woods in the winter at night"""},
                                {"role": "user", "content": message.choices[0].message.content}
                            ]
                        )
                        return f'a grunge digital painting of{message2.choices[0].message.content}, the subject is in the center of the image'
                    except Exception as e:
                        print(f"Error in thumbnail summary generation (attempt {attempt + 1}): {str(e)}")
                        if attempt < max_retries - 1:
                            wait_time = initial_delay * (2 ** attempt)
                            print(f"Retrying thumbnail summary in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        raise
                else:
                    # Second summary attempt for scenes using OpenAI
                    try:
                        message2 = oai_client.chat.completions.create(
                            model='chatgpt-4o-latest',
                            max_tokens=4000,
                            temperature=0.7,
                            messages=[
                                {"role": "system",
                                "content": f"""Write a very short description of a scene that shows what the given scene is about.
                                Only describe the setting of the scene.

                                ## Instructions
                                Do not write any disturbing details such as gore, blood, or violence of any kind.
                                Do not mention any children or people in the description, only describe the setting.
                                Do not mention any weapons in the description.
                                Do not describe any people being injured or killed.
                                Write this in 20 words or less.
        
                                ## Examples:
                                a person sitting on the edge of a bed in a dark concrete room
                                a young girl standing in a playground at night
                                a man standing outside a house at night
                                a scary person in a diner at night
                                a large sea creature in the dark ocean
                                a forest ranger standing in front of a large creature at night
                                an abandoned mansion at night
                                a cop car in a cornfield at night
                                a lighthouse on a cliff overlooking the ocean at night
                                an old graveyard in a forest at night
                                a wooden fire lookout tower at night
                                a cabin in the woods in the winter at night"""},
                                {"role": "user", "content": message.choices[0].message.content}
                            ]
                        )
                        return f'a film photograph of {message2.choices[0].message.content}'
                    except Exception as e:
                        print(f"Error in scene summary generation (attempt {attempt + 1}): {str(e)}")
                        if attempt < max_retries - 1:
                            wait_time = initial_delay * (2 ** attempt)
                            print(f"Retrying scene summary in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        raise

            else:
                prompt = 'portrait of a beautiful woman in a dress outside centered in the frame'
                return prompt

        except Exception as e:
            print(f"Error in initial summary generation (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt)
                print(f"Retrying initial summary in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            raise

    raise Exception(f"Failed to generate summary after {max_retries} attempts")


def generate_img(prompt, is_thumbnail=False, max_retries=5, initial_delay=1):
    if not replicate_client:
        raise ValueError("Replicate client not initialized")
    
    for attempt in range(max_retries):
        try:
            output = replicate_client.run(
                "black-forest-labs/flux-1.1-pro-ultra",
                input={
                    "prompt": prompt,
                    "aspect_ratio": "21:9" if is_thumbnail else "16:9",
                    "output_format": "png", 
                    "output_quality": 80,
                    "safety_tolerance": 5,
                    "prompt_upsampling": True
                }
            )
            return output
            
        except (ReplicateError, Exception) as e:
            wait_time = initial_delay * (2 ** attempt)  # Exponential backoff
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                raise
    
    raise Exception(f"Failed to generate image after {max_retries} attempts")

def save_image(image_url, profile_name, image_type='scene', scene_number=None):
    response = requests.get(image_url)
    if response.status_code == 200:
        if image_type == 'scene':
            directory = f"Output/{profile_name}/scene_images"
            filename = f"{directory}/scene_{scene_number}.png"
        elif image_type == 'thumbnail':
            directory = f"Output/{profile_name}"
            filename = f"{directory}/video_background.png"
        else:
            raise ValueError("Invalid image_type. Must be 'scene' or 'thumbnail'.")

        os.makedirs(directory, exist_ok=True)

        # Open the image from the response content
        image = Image.open(BytesIO(response.content))

        # First, resize the image to be 1080 pixels tall while maintaining aspect ratio
        aspect_ratio = image.width / image.height
        new_height = 1080
        new_width = int(new_height * aspect_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        if image_type == 'thumbnail':
            # For thumbnails, calculate the crop after resizing
            target_ratio = 1920 / 1080
            crop_width = int(new_height * target_ratio)
            
            # If image is wider than needed, crop from right side
            if new_width > crop_width:
                image = image.crop((0, 0, crop_width, new_height))
            else:
                # If image is not wide enough, resize to exact dimensions
                image = image.resize((1920, 1080), Image.LANCZOS)
        else:
            # For scene images, resize to exact dimensions
            image = image.resize((1920, 1080), Image.LANCZOS)
        
        # Save the image
        image.save(filename, format='PNG')

        return filename
    else:
        print(f"Failed to download the image {'for scene ' + str(scene_number) if image_type == 'scene' else 'for thumbnail'}")
        return None


def generate_tts(scene, voice_id, profile_name, scene_number):
    MAX_RETRIES = 25
    MAX_FILE_RETRIES = 3

    audiofiles_dir = f'Output/{profile_name}/audiofiles/scene_{scene_number}'
    os.makedirs(audiofiles_dir, exist_ok=True)
    output_path = f'{audiofiles_dir}/scene_{scene_number}.mp3'

    def verify_audio_file(file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            if len(audio) == 0:
                return False
            return True
        except Exception as e:
            print(f"Audio verification failed: {e}")
            return False

    def generate_single_audio(text_content):
        return elevenlabs_client.text_to_speech.convert(
            text=text_content,
            voice_id=voice_id,
            model_id='eleven_turbo_v2',
            output_format="mp3_44100_192",
            voice_settings=VoiceSettings(
                stability=0.50,
                similarity_boost=0.70,
                style=0.01,
                use_speaker_boost=True
            ),
        )

    # Split scene into paragraphs and group them
    paragraphs = [p.strip() for p in scene.split('\n') if p.strip()]
    paragraph_groups = [paragraphs[i:i+5] for i in range(0, len(paragraphs), 5)]
    
    # Generate audio for each paragraph group
    group_audio_files = []
    
    for group_idx, paragraph_group in enumerate(paragraph_groups):
        group_text = '\n'.join(paragraph_group)
        group_output_path = f'{audiofiles_dir}/group_{group_idx}.mp3'
        
        file_retries = 0
        success = False
        
        while file_retries < MAX_FILE_RETRIES and not success:
            retries = 0
            while retries < MAX_RETRIES and not success:
                try:
                    temp_path = f'{audiofiles_dir}/temp_{group_idx}.mp3'
                    
                    # Clean up any existing temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Generate new audio
                    audio = generate_single_audio(group_text)
                    
                    # Save to temporary file
                    with open(temp_path, "wb") as f:
                        for chunk in audio:
                            if chunk:
                                f.write(chunk)

                    # Verify the audio file
                    if verify_audio_file(temp_path):
                        # If verification succeeds, move to final location
                        if os.path.exists(group_output_path):
                            os.remove(group_output_path)
                        os.rename(temp_path, group_output_path)
                        group_audio_files.append(group_output_path)
                        success = True
                        break
                    else:
                        print(f"Generated audio file failed verification - will retry")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        retries += 1

                except Exception as e:
                    print(f"Error generating audio file: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    retries += 1
                    if retries < MAX_RETRIES:
                        print(f"Retrying... (attempt {retries + 1}/{MAX_RETRIES})")
                        time.sleep(1)

            if not success:
                file_retries += 1
                if file_retries < MAX_FILE_RETRIES:
                    print(f"Starting new generation attempt ({file_retries + 1}/{MAX_FILE_RETRIES})")
                else:
                    raise RuntimeError(f"Failed to generate valid audio file after {MAX_FILE_RETRIES} complete attempts")

    if not group_audio_files:
        raise RuntimeError(f"Failed to generate any valid audio files")
    
    # Combine all group audio files with silence between them
    silence = AudioSegment.silent(duration=600)  # 0.6 seconds
    combined_audio = AudioSegment.empty()
    
    for audio_file in group_audio_files:
        audio_segment = AudioSegment.from_file(audio_file)
        combined_audio += audio_segment + silence
    
    # Remove the last silence
    combined_audio = combined_audio[:-600]
    
    # Save the combined audio
    combined_audio.export(output_path, format='mp3')
    
    # Clean up individual group files
    for file_path in group_audio_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Get final duration
    duration = len(combined_audio)
    
    return output_path, duration

def combine_audio(profile_name, scene_number):
    audiofiles_dir = f'Output/{profile_name}/audiofiles/scene_{scene_number}'
    # Get a sorted list of all audio files in the directory
    files = os.listdir(audiofiles_dir)
    audio_files = natsorted(files)

    # Load the silence audio file
    silence = AudioSegment.silent(800)

    # Create a list to hold all of our audio segments
    segments = []

    # Iterate over the audio files
    for audio_file in audio_files:
        # Load the current audio file
        audio = AudioSegment.from_file(os.path.join(audiofiles_dir, audio_file))
        # Add the current audio file and silence to our list of segments
        segments.append(audio)
        segments.append(silence)

    # Remove the last silence segment from the list
    segments = segments[:-1]

    # Combine all audio segments
    combined = sum(segments)

    # Export the combined audio file
    output_audio_path = f'Output/{profile_name}/Scene_{scene_number}.mp3'
    combined.export(output_audio_path, format='mp3')
    
    shutil.rmtree(audiofiles_dir)

    return output_audio_path, len(combined)


def draw_text_on_image(image_path, text, output_path, profile):
    # Convert the text to uppercase
    text = text.upper()

    # Split the text into two parts
    if ',' in text:
        # Split after the last comma
        before_split, after_split = text.rsplit(', ', 1)
        before_split += ','
    elif '.' in text:
        before_split, after_split = text.split('. ', 1)
    elif ' AND ' in text:
        before_split, after_split = text.split(' AND ', 1)
        after_split = 'AND ' + after_split
    elif ' THAT ' in text:
        before_split, after_split = text.split(' THAT ', 1)
        after_split = 'THAT ' + after_split
    else:
        before_split = text
        after_split = ''

    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Define the bounding box
    left_padding = 20
    top_padding = 20
    bottom_padding = 100
    box_width = int(img_width * 0.7) - left_padding
    max_height = img_height - top_padding - bottom_padding

    # Set initial font size and define the bounding box
    font_size = 250  # Start with a large font size
    font_path = f'fonts/{profile["thumbnail"]["font"]}'  # Get font from channel settings
    font = ImageFont.truetype(font_path, font_size)

    # Create a separate image layer for the shadow
    shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)

    # Calculate the maximum font size that fits within the image height minus padding
    while True:
        # Wrap text separately for before and after the split
        wrapped_before_split = textwrap.fill(before_split, width=int(box_width / font.getbbox('A')[2]))
        wrapped_after_split = textwrap.fill(after_split, width=int(box_width / font.getbbox('A')[2]))

        # Calculate text size using textbbox
        combined_text = wrapped_before_split + '\n' + wrapped_after_split if after_split else wrapped_before_split
        text_bbox = shadow_draw.textbbox((0, 0), combined_text, font=font)
        text_height = text_bbox[3] - text_bbox[1]

        # Check if the text fits within the image height minus padding
        if text_height <= max_height:
            break
        else:
            # Reduce font size and try again
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)

    # Calculate the position to start the text, from the top with padding
    x = left_padding
    y = top_padding

    # Ensure the text does not exceed the image's bottom
    if y + text_height > img_height - bottom_padding:
        y = img_height - bottom_padding - text_height

    # Draw the drop shadow for both parts
    shadow_offset = (5, 5)  # Offset of the shadow (right and down)
    shadow_draw.text((x + shadow_offset[0], y + shadow_offset[1]), wrapped_before_split, font=font, fill="black",
                     align="left")

    if after_split:
        y_after_split = y + shadow_draw.textbbox((0, 0), wrapped_before_split, font=font)[3]
        shadow_draw.text((x + shadow_offset[0], y_after_split + shadow_offset[1]), wrapped_after_split, font=font,
                         fill="black", align="left")

    # Apply a Gaussian blur to the shadow layer to soften the shadow
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=30))

    # Paste the shadow layer onto the original image
    image = Image.alpha_composite(image.convert("RGBA"), shadow_layer)

    # Draw the text with a black stroke on the original image
    draw = ImageDraw.Draw(image)
    stroke_width = 8  # Width of the stroke

    # Draw the part before the split
    for offset in [(stroke_width, stroke_width), (-stroke_width, -stroke_width), (stroke_width, -stroke_width),
                   (-stroke_width, stroke_width)]:
        draw.text((x + offset[0], y + offset[1]), wrapped_before_split, font=font, fill="black", align="left")
    draw.text((x, y), wrapped_before_split, font=font, fill=profile['thumbnail']['primary_color'], align="left")

    # Draw the part after the split (colored differently)
    if after_split:
        y_after_split = y + draw.textbbox((0, 0), wrapped_before_split, font=font)[3]
        for offset in [(stroke_width, stroke_width), (-stroke_width, -stroke_width), (stroke_width, -stroke_width),
                       (-stroke_width, stroke_width)]:
            draw.text((x + offset[0], y_after_split + offset[1]), wrapped_after_split, font=font, fill="black",
                      align="left")
        draw.text((x, y_after_split), wrapped_after_split, font=font, fill=profile['thumbnail']['secondary_color'],
                  align="left")

    # Save the image as JPEG with the specified quality
    quality = 100  # Start with high quality
    image = image.convert("RGB")  # Convert to RGB mode as JPEG doesn't support transparency
    image.save(output_path, format="JPEG", quality=quality, optimize=True)

    # Reduce quality until the file size is less than the maximum allowed size
    max_file_size = 1.5 * 1024 * 1024  # Max file size in bytes
    while os.path.getsize(output_path) > max_file_size and quality > 10:
        quality -= 5
        image.save(output_path, format="JPEG", quality=quality, optimize=True)

    print(f"Image saved at {output_path} with final quality: {quality}")

def createTitles(story_text, finetune_model, max_retries=10):
    """Create a title with retry logic to ensure it meets criteria:
    - Must be between 70 and 100 characters
    - Must include a comma
    """
    while True:  # Keep generating titles until user accepts one
        for attempt in range(max_retries):
            title = oai_client.chat.completions.create(
                model=finetune_model,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": "You are tasked with creating a YouTube title for the given story. The title must be between 70 and 100 characters and include a comma. The title must be told in first person in the past tense."},
                    {"role": "user", "content": story_text},
                ]
            )
            
            title_text = title.choices[0].message.content.replace('"', '')
            
            # Add comma if missing (for horror stories)
            if 'Horror' in story_text and ',' not in title_text:
                title_text = title_text.replace(' ', ', ', 1)  # Add a comma after the first space
            
            # Check if title meets all criteria
            if len(title_text) <= 100 and len(title_text) >= 70 and ',' in title_text:
                # Use ANSI escape codes for red text
                print(f"\033[91mGenerated title: {title_text}")
                user_input = input("Accept this title? (y/n): \033[0m").lower()
                
                if user_input == 'y':
                    print(f"Title accepted: {title_text}")
                    return title_text
                else:
                    print("Generating new title...")
                    break  # Break inner loop to generate new title
            else:
                issues = []
                if len(title_text) > 100:
                    issues.append("too long")
                if ',' not in title_text:
                    issues.append("missing comma")
                print(f"Title invalid ({', '.join(issues)}) on attempt {attempt + 1}, retrying...")
    
        # If we've exhausted max_retries without finding a valid title
        if attempt == max_retries - 1:
            print(f"Warning: Could not generate valid title after {max_retries} attempts. Truncating...")
            return title_text[:97] + "..."

def crop_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Target dimensions
        target_width, target_height = 1920, 1080

        # Calculate aspect ratios
        img_aspect = img.width / img.height
        target_aspect = target_width / target_height

        if img_aspect > target_aspect:
            # Image is wider than target, crop width
            new_height = img.height
            new_width = int(new_height * target_aspect)
        else:
            # Image is taller than target, crop height
            new_width = img.width
            new_height = int(new_width / target_aspect)

        # Crop the center of the image
        left = (img.width - new_width) // 2
        top = (img.height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        img_cropped = img.crop((left, top, right, bottom))

        # Resize to target dimensions
        img_resized = img_cropped.resize((target_width, target_height), Image.LANCZOS)

        # Save the result
        img_resized.save(output_path)

    print(f"Image cropped and resized to 1920x1080 at {output_path}")


def overlay_image(background_path, overlay_path, output_path):
    try:
        with Image.open(background_path) as background:
            with Image.open(overlay_path) as overlay:
                # Ensure both images are in RGBA mode
                background = background.convert('RGBA')
                overlay = overlay.convert('RGBA')

                # Resize overlay to fit the background
                overlay_resized = overlay.resize(background.size)

                try:
                    # Try the original paste method
                    background.paste(overlay_resized, (0, 0), overlay_resized)
                except ValueError:
                    # If paste fails, use alpha_composite
                    background = Image.alpha_composite(background, overlay_resized)

                # Save the result
                background.save(output_path, 'PNG')

    except Exception as e:
        print(f"Error in overlay_image: {e}")
        # If all else fails, just use the background image
        Image.open(background_path).save(output_path)


def prepare_images(original_image_path, profile_name, overlay_image_path):
    output_dir = f'Output/{profile_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Prepare thumbnail background (left-aligned)
    thumbnail_bg_path = f'{output_dir}/thumbnail_bg.png'

    # Overlay image on the left-aligned thumbnail background
    thumbnail_path = f'{output_dir}/thumbnail_bg_final.png'
    overlay_image(thumbnail_bg_path, overlay_image_path, thumbnail_path)

    return thumbnail_path


def draw_text_on_image_with_settings(image_path, text, output_path, profile):
    # Convert the text to uppercase
    text = text.upper()

    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Define the bounding box
    left_padding = 20
    top_padding = 20
    bottom_padding = 100
    box_width = int(img_width * 0.7) - left_padding
    max_height = img_height - top_padding - bottom_padding

    # Set initial font size and define the bounding box
    font_size = 250
    font_path = f'fonts/{profile["thumbnail"]["font"]}'  # Get font from channel settings
    font = ImageFont.truetype(font_path, font_size)

    # Create a separate image layer for the shadow
    shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)

    # Calculate the maximum font size that fits within the image height minus padding
    while True:
        # Wrap text
        wrapped_text = textwrap.fill(text, width=int(box_width / font.getbbox('A')[2]))

        # Calculate text size using textbbox
        text_bbox = shadow_draw.textbbox((0, 0), wrapped_text, font=font)
        text_height = text_bbox[3] - text_bbox[1]

        # Check if the text fits within the image height minus padding
        if text_height <= max_height:
            break
        else:
            # Reduce font size and try again
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)

    # Calculate the position to start the text, from the top with padding
    x = left_padding
    y = top_padding

    # Ensure the text does not exceed the image's bottom
    if y + text_height > img_height - bottom_padding:
        y = img_height - bottom_padding - text_height

    # Draw the drop shadow
    shadow_offset = (5, 5)  # Offset of the shadow (right and down)
    shadow_draw.text((x + shadow_offset[0], y + shadow_offset[1]), wrapped_text, font=font, fill="black", align="left")

    # Apply a Gaussian blur to the shadow layer to soften the shadow
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=30))

    # Paste the shadow layer onto the original image
    image = Image.alpha_composite(image.convert("RGBA"), shadow_layer)

    # Draw the text with a stroke on the original image
    draw = ImageDraw.Draw(image)
    stroke_width = profile['thumbnail']['stroke_width']
    stroke_color = profile['thumbnail']['stroke_color']

    # Split the wrapped text into lines
    lines = wrapped_text.split('\n')

    # Draw each line with appropriate coloring
    current_y = y
    for i, line in enumerate(lines):
        # Draw the stroke
        for offset in [(stroke_width, stroke_width), (-stroke_width, -stroke_width), (stroke_width, -stroke_width),
                       (-stroke_width, stroke_width)]:
            draw.text((x + offset[0], current_y + offset[1]), line, font=font, fill=stroke_color, align="left")

        # Determine coloring method
        if profile['thumbnail']['color_method'] == 'alternate_lines':
            color = profile['thumbnail']['primary_color'] if i % 2 == 0 else profile['thumbnail']['secondary_color']
            draw.text((x, current_y), line, font=font, fill=color, align="left")
        else:
            draw.text((x, current_y), line, font=font, fill=profile['thumbnail']['primary_color'], align="left")

        # Move to the next line with 120% spacing
        line_height = draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
        current_y += int(line_height * 1.2)  # 120% of the line height

    # Save the image as JPEG with the specified quality
    quality = 100  # Start with high quality
    image = image.convert("RGB")  # Convert to RGB mode as JPEG doesn't support transparency
    image.save(output_path, format="JPEG", quality=quality, optimize=True)

    # Reduce quality until the file size is less than the maximum allowed size
    max_file_size = 1.5 * 1024 * 1024  # Max file size in bytes
    while os.path.getsize(output_path) > max_file_size and quality > 10:
        quality -= 5
        image.save(output_path, format="JPEG", quality=quality, optimize=True)

    print(f"Image saved at {output_path} with final quality: {quality}")


def create_thumbnail(title, channel_name, background_image, username):
    """Create thumbnail with channel-specific settings"""
    # No need to get channel settings separately as they're already initialized
    thumbnail_output_path = f'Output/{channel_name}/thumbnail.png'

    if settings.COLOR_METHOD == 'after_punctuation':
        draw_text_on_image(background_image, title, thumbnail_output_path, {
            'thumbnail': {
                'font': settings.THUMBNAIL_FONT,
                'primary_color': settings.THUMBNAIL_PRIMARY_COLOR,
                'secondary_color': settings.THUMBNAIL_SECONDARY_COLOR,
                'stroke_color': settings.THUMBNAIL_STROKE_COLOR,
                'stroke_width': settings.THUMBNAIL_STROKE_WIDTH
            }
        })
    else:
        draw_text_on_image_with_settings(background_image, title, thumbnail_output_path, {
            'thumbnail': {
                'font': settings.THUMBNAIL_FONT,
                'primary_color': settings.THUMBNAIL_PRIMARY_COLOR,
                'secondary_color': settings.THUMBNAIL_SECONDARY_COLOR,
                'stroke_color': settings.THUMBNAIL_STROKE_COLOR,
                'stroke_width': settings.THUMBNAIL_STROKE_WIDTH,
                'color_method': settings.COLOR_METHOD
            }
        })

    print(f"Thumbnail saved to {thumbnail_output_path}")
    return thumbnail_output_path


def process_local(username: str, channel_name: str, story_text=None, scenes=None, story_idea=None, use_existing_audio=False):
    print("Starting local processing...")
    # Initialize settings and clients
    settings.initialize_settings(username)
    settings.initialize_channel_settings(username, channel_name)
    initialize_clients(username)
    
    # Create title using channel's fine-tuned model
    title = createTitles(story_idea, settings.STORY_TITLE_FT_MODEL)
    title = title.replace('\"', '')
    print(f'Video Title: {title}')

    # Generate thumbnail image
    thumbnail_summary = summarize_text(story_text, settings.STORY_PROFILE, is_thumbnail=True)
    thumbnail_image_url = generate_img(thumbnail_summary, is_thumbnail=True)
    thumbnail_bg_path = save_image(thumbnail_image_url, channel_name, image_type='thumbnail')

    # Create thumbnail using channel settings
    thumbnail_path = create_thumbnail(title, channel_name, thumbnail_bg_path, username)

    # Use video_background.png for all scenes
    scene_images = [f'Output/{channel_name}/video_background.png'] * len(scenes)
    
    # Process each scene
    scene_audio_files = []
    scene_durations = []
    final_audio_path = f'Output/{channel_name}/Final.mp3'

    if not use_existing_audio:
        print("\nProcessing scenes...")
        for i, scene in tqdm(enumerate(scenes, 1), total=len(scenes), desc="Processing Scenes"):
            audio_file, duration = generate_tts(
                scene, 
                settings.VOICE_ID,
                channel_name, 
                i
            )
            scene_audio_files.append(audio_file)
            scene_durations.append(duration)

        # Combine audio files
        final_audio = AudioSegment.empty()
        silence = AudioSegment.silent(duration=600)
        for audio_file in scene_audio_files:
            final_audio += AudioSegment.from_mp3(audio_file) + silence
        final_audio = final_audio[:-600]  # Remove last silence
        
        final_audio.export(final_audio_path, format='mp3')
    else:
        print("\nUsing existing Final.mp3 file...")
        if not os.path.exists(final_audio_path):
            raise FileNotFoundError(f"Could not find existing audio file at {final_audio_path}")
        # Calculate duration from existing file
        audio = AudioSegment.from_mp3(final_audio_path)
        scene_durations = [len(audio)]  # Total duration in milliseconds
        scene_audio_files = [final_audio_path]

    print("Local processing completed.")
    return scene_images, scene_audio_files, scene_durations, final_audio_path, thumbnail_path, title

if __name__ == "__main__":
    settings.initialize_settings('229202')
    initialize_clients()

