import praw
import random
import os
import re
from openai import OpenAI
from collections import deque
from typing import List, Dict
from tqdm import tqdm
import settings
import json
import praw

# Add global variables to hold the clients
oai_client = None
or_client = None
reddit = None

def initialize_clients():
    """Initialize API clients with credentials"""
    global oai_client, or_client, reddit
    
    api_key = settings.OAI_API_KEY
    
    if not api_key:
        raise ValueError("OpenAI API key is empty or None")
    
    # Initialize OpenAI client
    try:
        oai_client = OpenAI(api_key=api_key)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        raise
        
    # Initialize OpenRouter client
    try:
        if settings.OR_API_KEY:
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=settings.OR_API_KEY
            )
            print("OpenRouter client initialized successfully")
        else:
            or_client = None
            print("Skipping OpenRouter client initialization (no API key)")
    except Exception as e:
        print(f"Error initializing OpenRouter client: {str(e)}")
        or_client = None
        
    # Initialize Reddit client
    try:
        reddit = praw.Reddit(
            client_id='1oQZd_uYc9Wl7Q',
            client_secret='uanzrHod7xZya1VSZ2ZTzEVXnlA',
            user_agent="Reddit posts"
        )
        print("Reddit client initialized successfully")
    except Exception as e:
        print(f"Error initializing Reddit client: {str(e)}")
        reddit = None
    
    print("Client initialization completed")

# previous_scenes = deque(maxlen=2)
previous_scenes = deque(maxlen=4)


def write_detailed_scene_description(scene: str) -> str:
    prompt = f"""
    Analyze the following scene and provide a highly detailed paragraph focusing on the most important details and events that are crucial to the story.
    You must include every single detail exactly that is most important to the plot of the story.
    
    Be as detailed as possible with your description of the events in the scene, your description must be at least 200 words.
    Do not write any comments, only return the description.

    Scene:
    {scene}

    Provide the description as a structured text, not in JSON format.
    """
    retries = 0
    while retries < 5:
        try:
            response = or_client.chat.completions.create(
                model=settings.OR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error {e}. Retrying...")
            retries += 1


def check_scene_consistency(new_scene_description: str, previous_scenes: List[str]) -> str:
    prompt = f"""
    You are an expert story editor, compare the new scene description with the previous scenes and identify any continuity errors that are crucial to the progression the story.
    
    ## Ignore new elements or changes if they make sense in the context of the story progressing, do not label something an error just because it's different from the previous scene.
    Your job is to make sure the story flows coherently and makes sense.

    ##Only provide fixes that can be fixed in the given scene, do not provide fixes that are for anything that could be fixed in a previous scene.

    Only write the most important continuity errors about the plot, characters, and story timeline.
    If you find continuity errors then respond with them in a list from most important to least important with the most important ones labeled that.
    Ignore any minor continuity errors that are there for the progression of the story or are minor details in the story, you are only looking for the most important details that are crucial to the plot of the story.
    Only respond with the list of continuity errors, do not write any comments.
    If you find no continuity errors with the previous scenes then only respond with: No Continuity Errors Found.

    New scene:
    {new_scene_description}

    Previous scenes (most recent 4):
    {' '.join(previous_scenes)}

    Provide the continuity errors as a list in order of importance to the story. Describe how to fix those errors in the scene.
    """

    response = or_client.chat.completions.create(
        model=settings.OR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    print(f'########### Errors #############\n\n{response.choices[0].message.content}\n\n##############################')
    return response.choices[0].message.content


def rewrite_scene(original_scene: str, scene_beat: str, inconsistencies: str) -> str:
    print("Rewriting scene to address inconsistencies...")
    prompt = f"""
    Rewrite the following scene to address the identified inconsistencies while maintaining the original scene beat. Focus on fixing the most important inconsistencies first.

    Original scene:
    {original_scene}

    Scene beat:
    {scene_beat}

    Issues to address:
    {inconsistencies}

    Rewrite the scene to maintain story continuity and address these issues. Make sure to resolve ALL inconsistencies in your rewrite.
    The rewrite should maintain the same general length and level of detail as the original scene.
    """

    response = or_client.chat.completions.create(
        model=settings.OR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    print("Scene rewritten.")
    return response.choices[0].message.content


def write_story(outline, characters, add_transitions=False):
    print("Starting story writing process...")
    
    scenes = []
    edited_scenes = []
    original_scenes = []

    total_steps = len(outline) * 2  # Writing and editing for each scene
    progress_bar = tqdm(total=total_steps, desc="Overall Progress", unit="step")

    # Initialize as None before the loop
    next_scene = None
    
    for num, scene_beat in enumerate(outline):
        # If we already wrote this scene as next_scene, use it
        if next_scene:
            scene = next_scene
        else:
            # Write scene (only happens for first scene)
            progress_bar.set_description(f"Writing Scene {num+1}")
            scene = write_scene(scene_beat, characters, num, len(outline))
            
        previous_scenes.append(scene)
        original_scenes.append(scene)
        
        # Add transition if enabled and not the last scene
        if add_transitions and num < len(outline) - 1:
            # Write the next scene (will be used in next iteration)
            next_scene = write_scene(outline[num + 1], characters, num + 1, len(outline))
            transition = write_scene_transition(scene, next_scene)
            print(f"Transition: {transition}")
            scene = f"{scene}\n\n{transition}"
        else:
            next_scene = None
        
        scenes.append(scene)
        progress_bar.update(1)

    # Second pass: Edit all scenes with transitions included
    for num, scene in enumerate(scenes):
        progress_bar.set_description(f"Editing Scene {num+1}")
        edited_scene = callTune4(scene)
        edited_scenes.append(edited_scene)
        progress_bar.update(1)

    progress_bar.close()

    final_story = '\n\n'.join(edited_scenes)

    return final_story, edited_scenes, original_scenes


def write_scene(scene_beat: str, characters: str, num, total_scenes) -> str:
    print(f'Writing scene {num+1} of {total_scenes}\n\n{scene_beat}\n\n')
    # Get only the 4 most recent scenes for context
    recent_context = list(previous_scenes)[-2:] if previous_scenes else ["No previous context. This is the first scene."]
    context = '\n\n'.join(recent_context)

    final_scene_indicator = 'This is the final scene of the story. You must write an ending to the story that nicely ends the story explicitly, do not end it in the middle of a scene or event. Do not write "The End" or anything like that.' if num == total_scenes - 1 else ''

    prompt = f"""
    ## SCENE CONTEXT AND CONTINUITY
    # Characters
    {characters}
    
    # Use the provided STORY CONTEXT to remember details and events from the previous scenes in order to maintain consistency in the new scene you are writing.
    ## STORY CONTEXT
    {context}
    
    # Scene Beat to Write
    {scene_beat}

    ## WRITING INSTRUCTIONS
    You are an expert fiction writer. Write a fully detailed scene as long as you need to without overwriting that flows naturally from the previous events described in the context.
    {final_scene_indicator}

    # Core Requirements
    - Write from first-person narrator perspective only
    - Begin with a clear connection to the previous scene's ending
    - Include full, natural dialogue
    - Write the dialogue in their own paragraphs, do not include the dialogue in the same paragraph as the narration.
    - Write everything that the narrator sees, hears, and everything that happens in the scene.
    - Write the entire scene and include everything in the scene beat given, do not leave anything out.
    
    # Pacing and Suspense
    - Maintain steady, escalating suspense
    - Use strategic pauses and silence for impact
    - Build tension in small, deliberate increments
    - Balance action with reflection

    # Writing Style
    - Use concise, sensory-rich language
    - Vary sentence length based on tension:
        * Shorter sentences for action/tension
        * Longer sentences for introspection
    - Show emotions through implications rather than stating them
    
    # Scene Structure
    - Write tight, focused paragraphs
    - Layer the scene from normal to unsettling
    - Break up dialogue with introspection and description
    - Include moments of dark humor sparingly
    - Allow for natural processing of events
    """
    
    retries = 0
    while retries < 5:
        try:
            response = or_client.chat.completions.create(
                model=settings.OR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )

            written_scene = response.choices[0].message.content.replace('*', '').replace('---\n', '').replace('\n\n---', '')
            
            # Check if scene is too short (less than 500 characters)
            if len(written_scene.strip()) < 500:
                print(f"Scene too short ({len(written_scene.strip())} chars). Retrying...")
                retries += 1
                continue
                
            print(written_scene)
            
            # Add scene consistency check with verification loop
            max_attempts = 3
            attempt = 0
            
            while attempt < max_attempts:
                # Get detailed description and check consistency
                detailed_scene = write_detailed_scene_description(written_scene)
                inconsistencies = check_scene_consistency(detailed_scene, [write_detailed_scene_description(prev) for prev in previous_scenes])
                
                # If no inconsistencies found, return the scene
                if not inconsistencies or "No Continuity Errors Found" in inconsistencies:
                    return written_scene
                    
                # Rewrite scene to fix inconsistencies
                print(f"Attempt {attempt + 1}: Rewriting scene to fix inconsistencies...")
                written_scene = rewrite_scene(written_scene, scene_beat, inconsistencies)
                
                # Verify the fixes and rewrite if needed
                verification_result = verify_scene_fixes(written_scene, inconsistencies)
                if verification_result == "All issues resolved":
                    return written_scene
                else:
                    # Rewrite again with remaining issues
                    written_scene = rewrite_scene(written_scene, scene_beat, verification_result)
                    
                attempt += 1
                print(f"Verification failed. Remaining issues: {verification_result}")
            
            print("Warning: Maximum rewrite attempts reached. Using best version.")
            return written_scene
            
        except Exception as e:
            print(f"Error {e}. Retrying...")
            retries += 1
    
    raise Exception("Failed to write scene after 5 attempts")

def verify_scene_fixes(rewritten_scene: str, original_issues: str) -> str:
    prompt = f"""
    Verify if the rewritten scene has properly addressed all the previously identified issues.
    
    Original issues to fix:
    {original_issues}

    Rewritten scene:
    {rewritten_scene}

    Check if each issue has been properly resolved. If any issues remain unresolved, list them specifically.
    Format remaining issues as a clear, numbered list that can be used for another rewrite.
    If all issues are resolved, respond only with: All issues resolved
    """

    response = or_client.chat.completions.create(
        model=settings.OR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    verification_result = response.choices[0].message.content
    print(f"Verification result: {verification_result}")
    return verification_result

def find_long_post(story_profile):
    subreddit = reddit.subreddit(story_profile['subreddit'])
    long_posts = []

    output_folder = 'stories'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    history_file = f"{story_profile['subreddit']}_history.txt"
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            history_ids = file.read().splitlines()
    else:
        history_ids = []

    for post in subreddit.top(time_filter='year', limit=1000):
        if (post.id not in history_ids and
                post.link_flair_text != story_profile['flair_exclude'] and
                len(post.selftext) >= story_profile['min_length'] and
                "part" not in post.title.lower()):
            long_posts.append(post)

    if long_posts:
        selected_post = random.choice(long_posts)
        filename = os.path.join(output_folder, f"{story_profile['subreddit']}_{selected_post.title}.txt".replace('/', '_').replace('\\', '_'))
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(selected_post.selftext)

        with open(history_file, 'a') as file:
            file.write(f"{selected_post.id}\n")

        print(f"Post saved: {selected_post.title}")
        print(f"File Name: {filename}")
        print(f'Post Length: {len(selected_post.selftext)}')

        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        print(f"No suitable post found for {story_profile['subreddit']}.")
        return None


def story_ideas():
    # Get the story profile name from the settings
    story_profile_name = settings.STORY_PROFILE
    
    # Get the corresponding story profile from the MongoDB collection
    try:
        all_story_profiles = settings.load_story_profiles()
        story_profile = all_story_profiles.get(story_profile_name)
        
        if not story_profile:
            print(f"Error: Story profile '{story_profile_name}' not found in the video-types collection")
            return None
    except Exception as e:
        print(f"Error loading story profiles: {str(e)}")
        return None

    if settings.USE_REDDIT:
        return find_long_post(story_profile)
    elif settings.USE_FINE_TUNE:
        prompt = random.choice(story_profile['prompts'])
        print(f'\n\n{prompt}\n\n')
        message = oai_client.chat.completions.create(
            model=story_profile['model'],
            messages=[
                {
                    "role": "system",
                    "content": story_profile['system_prompt']
                },
                {
                    "role": "user",
                    "content": prompt
                }])
        
        return message.choices[0].message.content
    else:
        prompt = random.choice(story_profile['prompts'])
        message = oai_client.chat.completions.create(
            model='gpt-4o-2024-08-06',
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }])

        return message.choices[0].message.content


def format_scenes(input_string):
    try:
        # Clean up the input string
        # Remove markdown code block markers and 'json' language identifier
        input_string = re.sub(r'```json\s*|\s*```', '', input_string)
        
        # Remove any leading/trailing whitespace
        input_string = input_string.strip()
        
        # Try to parse the input as JSON
        scenes = json.loads(input_string)
        
        formatted_scenes = []
        
        # Process each scene in the JSON array
        for scene in scenes:
            scene_number = scene.get('scene_number')
            scene_beat = scene.get('scene_beat')
            
            if scene_number is not None and scene_beat:
                formatted_scenes.append(f"{scene_beat.strip()}")
        
        if not formatted_scenes:
            print("Warning: No scenes were parsed from the JSON")
            return None
            
        return formatted_scenes
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON ({str(e)})")
        
        try:
            # Try to clean up the JSON string more aggressively
            # Remove all newlines and extra spaces between JSON objects
            cleaned_input = re.sub(r'\s+', ' ', input_string)
            cleaned_input = re.sub(r',\s*]', ']', cleaned_input)
            
            scenes = json.loads(cleaned_input)
            formatted_scenes = []
            
            for scene in scenes:
                scene_number = scene.get('scene_number')
                scene_beat = scene.get('scene_beat')
                
                if scene_number is not None and scene_beat:
                    formatted_scenes.append(f"{scene_beat.strip()}")
            
            if not formatted_scenes:
                print("Warning: No scenes were parsed from the cleaned JSON")
                return None
                
            return formatted_scenes
            
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON even after cleanup: {str(e)}")
            return None


def create_outline(idea, num=12):
    num = random.randrange(6, 10)

    retries = 0
    while retries < 5:
        try:
            message = oai_client.chat.completions.create(
                model='gpt-4o-2024-11-20',
                temperature=1,
                messages=[
                    {"role": "user", "content":
                        f'''## Instructions
                        
                        Write a full plot outline for the given story idea.
                        Write the plot outline as a list of all the scenes in the story. Each scene must be a highly detailed paragraph on what happens in that scene.
                        Each scene beat must include as much detail as you can about the events that happen in the scene.
                        Explicitly state the change of time between scenes if necessary.
                        Mention any locations by name.
                        A scene in the story is defined as when there is a change in the setting in the story.
                        The plot outline must contain {num} scenes.
                        The plot outline must follow and word things in a way that are from the protagonist's perspective, do not write anything from an outside character's perspective that the protagonist wouldn't know.
                        Only refer to the protagonist in the story as "The Protagonist" in the plot outline.
                        Each scene must smoothly transition from the previous scene and to the next scene without unexplained time and setting jumps.
                        Ensure key story elements (e.g., character motivations, mysteries, and plot developments) are resolved by the end.
                        Explicitly address and resolve the purpose and origin of central objects or plot devices (e.g., mysterious items, symbols, or events).
                        If other characters have significant knowledge of the mystery or key events, show how and when they gained this knowledge to maintain logical consistency.
                        Explore and resolve character dynamics, especially those affecting key relationships (e.g., family tension or conflicts).
                        Provide clarity on thematic or mysterious elements that connect scenes, ensuring the stakes are clearly defined and resolved.
                        The final scene beat must state it's the final scene beat of the story and how to end the story.


                        ## You must use following json format for the plot outline exactly without deviation:
                        [
                            {{"scene_number": 1, "scene_beat": "<Write the first scene beat here>"}},
                            {{"scene_number": 2, "scene_beat": "<Write the second scene beat here>"}},
                            {{"scene_number": 3, "scene_beat": "<Write the third scene beat here>"}},
                            {{"scene_number": 4, "scene_beat": "<Write the fourth scene beat here>"}},
                            {{"scene_number": 5, "scene_beat": "<Write the fifth scene beat here>"}},
                            {{"scene_number": 6, "scene_beat": "<Write the sixth scene beat here>"}},
                            {{"scene_number": 7, "scene_beat": "<Write the seventh scene beat here>"}},
                            {{"scene_number": 8, "scene_beat": "<Write the eighth scene beat here>"}},
                            {{"scene_number": 9, "scene_beat": "<Write the ninth scene beat here>"}},
                            {{"scene_number": 10, "scene_beat": "<Write the tenth scene beat here>"}},
                            {{"scene_number": 11, "scene_beat": "<Write the eleventh scene beat here>"}},
                            {{"scene_number": 12, "scene_beat": "<Write the twelfth scene beat here>"}}
                        ]
                        \n\n## Story Idea:\n{idea}'''}
                ]
            )

            print(message.choices[0].message.content)

            outline = format_scenes(message.choices[0].message.content)
            
            if not outline:
                print("Error: Empty outline generated.")
                retries += 1
                continue

            return outline
        except Exception as e:
            print(f"Error in create_outline: {e}. Retrying...")
            retries += 1

    print("Failed to create outline after 5 attempts.")
    return None


def characters(outline):
    retries = 0
    while retries < 10:
        try:
            message = or_client.chat.completions.create(
                model=settings.OR_MODEL,
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content":
                        f"""## Instructions
                        
                        Using the given story outline, write short character descriptions for all the characters in the story in the following format:
                        <character name='(Character Name)' aliases='(Character Alias)'>(Character description)</character>

                        The character alias is what the other characters in the story will call that character in the story such as their first name.
                        For the Protagonist's alias you must create a name that other characters will call them in the story.
                        The character description must only describe their appearance and their personality DO NOT write what happens to them in the story.
                        Only return the character descriptions without any comments.
        
                        ## Outilne:\n\n{outline}"""}
                ]
            )
            return message.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            retries += 1


def cleanup_scene(scene):
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        completion = oai_client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": f"""## Instruction\nYou are tasked with identifying any unnecessary foreshadowing paragraphs at the end of the given scene. You must only respond with the number of paragraphs that fit that description as an integer and not a word.\n\n## Scene:\n{scene}""",
                },
            ],
        )
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            try:
                paragraphs = scene.split('\n\n')
                num_to_remove = int(completion.choices[0].message.content)
                if num_to_remove > 0:
                    paragraphs = paragraphs[:-num_to_remove]
                    result = '\n\n'.join(paragraphs)
                    return result
                else:
                    return scene
            except ValueError:
                print(f"Error: Invalid response from API. Retrying... (Attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
        else:
            print(f"Error: No valid response from API. Retrying... (Attempt {retry_count + 1}/{max_retries})")
            retry_count += 1
    
    print("Max retries reached. Returning original scene.")
    return scene

import re

def process_scene(scene):
    completion2 = or_client.chat.completions.create(
            model='anthropic/claude-3-5-haiku-20241022:beta',
            messages=[
                {
                    "role": "user",
                    "content": f'Separate the given scene into sections of narrative and dialogue. Group consecutive paragraphs of narrative and dialogue under the same group while maintaining the original order of the scene. Label the narrative as [Narrative] and the dialogue as [Dialogue]. Only return the scene with the dialogue and narrative grouped without any comments.\n\n## Scene:\n{scene}'
                }])

    scene_text = completion2.choices[0].message.content
    # Split the scene into segments while preserving order
    segments = []
    current_segment = []
    current_type = None
    
    # Split text into lines
    lines = scene_text.split('\n')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Check if line starts with dialogue markers
        is_dialogue = line.strip().startswith('"')
        
        # Determine segment type
        if '[Narrative]' in line:
            if current_segment:
                segments.append((current_type, '\n'.join(current_segment)))
                current_segment = []
            current_type = 'narrative'
        elif '[Dialogue]' in line:
            if current_segment:
                segments.append((current_type, '\n'.join(current_segment)))
                current_segment = []
            current_type = 'dialogue'
        elif line.strip():  # If line is not empty
            current_segment.append(line.strip())
    
    # Add the last segment
    if current_segment:
        segments.append((current_type, '\n'.join(current_segment)))
    
    # Process segments based on their type
    processed_segments = []
    for segment_type, content in segments:
        if segment_type == 'narrative':
            # Send to narrative prompt (placeholder)
            processed_content = process_narrative(content)
        else:
            # Send to dialogue prompt (placeholder)
            processed_content = process_dialogue(content)
        processed_segments.append(processed_content)
    
    # Combine processed segments
    final_text = '\n\n'.join(processed_segments)
    return final_text

def process_narrative(narrative_text):
    completion = oai_client.chat.completions.create(
                model='ft:gpt-4o-2024-08-06:personal:jgrupe-narration-ft:AQnm6wr1',
                temperature=0.7,
                messages=[{
                    "role": "system",
                    "content": 'You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style.'},
                    {"role": "user",
                    "content": narrative_text}])
    return completion.choices[0].message.content

def process_dialogue(dialogue_text):
    completion = oai_client.chat.completions.create(
                model='ft:gpt-4o-2024-08-06:personal:jgrup-dialogue:ASBnHsCZ',
                temperature=0.7,
                messages=[{
                    "role": "system",
                    "content": 'You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style.'},
                    {"role": "user",
                    "content": dialogue_text}])
    return completion.choices[0].message.content

def callTune(scene):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:    
        try:
            completion2 = oai_client.chat.completions.create(
                model=settings.FT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style."
                    },
                    {
                        "role": "user",
                        "content": scene
                    }
                ]
            )
            
            rewritten_text = replace_words(completion2.choices[0].message.content)
            
            # Check if output is same as input
            if rewritten_text == scene:
                retry_count += 1
                continue

            return rewritten_text

        except Exception as e:
            print(f"Error in callTune: {e}")
            retry_count += 1
            
    # Return original scene if max retries reached
    return scene

def callTune2(scene):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        completion2 = oai_client.chat.completions.create(
            model=settings.FT_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style."
                },
                {
                    "role": "user",
                    "content": scene
                }])

        rewritten_text = completion2.choices[0].message.content
        
        # Check if output is same as input
        if rewritten_text == scene:
            print(f"Rewrite attempt {retry_count + 1} returned same text. Retrying...")
            retry_count += 1
            continue

        completion1 = or_client.chat.completions.create(
            model=settings.OR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""Remove all appositive phrases relating to people or objects in the given text, except those that contain foreshadowing.
Remove all absolute phrases relating to people or objects in the given text, except those that provide sensory information or describe physical sensations.
Remove all metaphors in the given text.
Remove any sentences that add unnecessary detail or reflection without contributing new information to the scene.

If a paragraph doesn't need to be changed then just leave it as is in the returned text.

Only respond with the modified text and nothing else.

## Text to edit:
{rewritten_text}"""}])

        return replace_words(completion1.choices[0].message.content), replace_words(rewritten_text)
        
    # If max retries reached, return original scene
    print("Max retries reached. Using original scene.")
    return replace_words(scene), replace_words(scene)


def callTune3(scene):
    completion1 = oai_client.chat.completions.create(
            model=settings.OAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""Remove all appositive phrases relating to people or objects in the given text, except those that contain foreshadowing.
Remove all absolute phrases relating to people or objects in the given text, except those that provide sensory information or describe physical sensations, or serve as contextual cues.
Remove all metaphors in the given text.
Remove any sentences that add unnecessary detail or reflection without contributing new information to the scene.

Seperate all of the dialogue into separate paragraphs.

Only respond with the modified text and nothing else.

## Text to edit:
{scene}"""}])

    return replace_words(completion1.choices[0].message.content)


def callTune4(scene):
    # Split into paragraphs
    paragraphs = [p.strip() for p in scene.split('\n\n') if p.strip()]
    
    # Group paragraphs based on if they contain quotes
    groups = []
    current_group = []
    contains_quote = None
    
    for p in paragraphs:
        current_contains_quote = '"' in p
        
        # Start new group if quote status changes
        if contains_quote is not None and current_contains_quote != contains_quote:
            groups.append('\n\n'.join(current_group))
            current_group = []
            
        current_group.append(p)
        contains_quote = current_contains_quote
    
    # Add final group
    if current_group:
        groups.append('\n\n'.join(current_group))
    
    # Process each group with appropriate fine-tuning
    processed_groups = []
    for group in groups:
        if '"' in group:
            processed_groups.append(group)
            # Process dialogue group
            # max_retries = 3
            # retry_count = 0
            # while retry_count < max_retries:
            #     completion = oai_client.chat.completions.create(
            #         model='ft:gpt-4o-2024-08-06:personal:jgrup-dialogue:ASBnHsCZ',
            #         temperature=0.7,
            #         messages=[{
            #             "role": "system", 
            #             "content": 'You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style.'},
            #             {"role": "user",
            #             "content": group}])
            #     output = completion.choices[0].message.content
                
            #     if len(output) <= len(group) * 2:
            #         processed_groups.append(output)
            #         break
            #     retry_count += 1
            #     if retry_count == max_retries:
            #         processed_groups.append(group) # Use original if all retries fail
        else:
            # Process narrative group 
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    completion = oai_client.chat.completions.create(
                        model='ft:gpt-4o-2024-08-06:personal:jgrupe-narration:AQl6Fs75',
                        temperature=0.7,
                        messages=[{
                            "role": "system", 
                            "content": 'You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style.'},
                            {"role": "user",
                            "content": group}])
                    output = completion.choices[0].message.content
                    
                    # Add success condition
                    if len(output) <= len(group) * 2:  # Similar to dialogue check
                        processed_groups.append(output)
                        break
                        
                    retry_count += 1
                    if retry_count == max_retries:
                        processed_groups.append(group)  # Use original if all retries fail
                        
                except Exception as e:
                    print(f"Error processing narrative group: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        processed_groups.append(group)
            
    # Combine processed groups and apply word replacements
    final_text = '\n\n'.join(processed_groups)
    return final_text.replace('*', '').replace('_', '')

def callTune5(scene):
    # Split into paragraphs
    paragraphs = [p.strip() for p in scene.split('\n\n') if p.strip()]
    
    # Group paragraphs based on if they start with quotes
    groups = []
    current_group = []
    starts_with_quote = None
    
    for p in paragraphs:
        current_starts_with_quote = p.lstrip().startswith('"')
        
        # Start new group if quote status changes
        if starts_with_quote is not None and current_starts_with_quote != starts_with_quote:
            groups.append('\n\n'.join(current_group))
            current_group = []
            
        current_group.append(p)
        starts_with_quote = current_starts_with_quote
    
    # Add final group
    if current_group:
        groups.append('\n\n'.join(current_group))
    
    # Process each group with appropriate fine-tuning
    processed_groups = []
    for group in groups:
        if group.lstrip().startswith('"'):
            # Process dialogue group
            # Process dialogue group
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                completion = oai_client.chat.completions.create(
                    model='ft:gpt-4o-2024-08-06:personal:jgrup-dialogue:ASBnHsCZ',
                    temperature=0.7,
                    messages=[{
                        "role": "system", 
                        "content": 'You are an expert copy editor tasked with re-writing the given text in Insomnia Stories unique voice and style.'},
                        {"role": "user",
                        "content": group}])
                output = completion.choices[0].message.content
                
                if len(output) <= len(group) * 2:
                    processed_groups.append(output)
                    break
                retry_count += 1
                if retry_count == max_retries:
                    processed_groups.append(group) # Use original if all retries fail
        else:
            # Process narrative group 
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    completion = or_client.chat.completions.create(
                        model=settings.OR_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Remove all appositive phrases relating to people or objects in the given text, except those that contain foreshadowing.
Remove all absolute phrases relating to people or objects in the given text, except those that provide sensory information or describe physical sensations.
Remove all metaphors in the given text.
Remove any sentences that add unnecessary detail or reflection without contributing new information to the scene.

If a paragraph doesn't need to be changed then just leave it as is in the returned text.

Only respond with the modified text and nothing else.

## Text to edit:
{group}"""}])
                    output = completion.choices[0].message.content
                    
                    # Add success condition
                    if len(output) <= len(group) * 2:  # Similar to dialogue check
                        processed_groups.append(output)
                        break
                        
                    retry_count += 1
                    if retry_count == max_retries:
                        processed_groups.append(group)  # Use original if all retries fail
                        
                except Exception as e:
                    print(f"Error processing narrative group: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        processed_groups.append(group)
            
    # Combine processed groups and apply word replacements
    final_text = '\n\n'.join(processed_groups)
    return final_text.replace('*', '').replace('_', '')

def replace_words(text):
    word_bank = {
    'shifted': 'moved',
    'shift': 'change',
    'shifting': 'changing',
    'bravado': 'bravery',
    'loomed': 'appeared',
    'visage': 'face',
    'abyssal': 'deep',
    'amidst': 'surrounded by',
    'amiss': 'wrong',
    'ancient': 'old',
    'abruptly': 'suddenly',
    'awash': 'covered',
    'apprehension': 'dread',
    'beacon': 'signal',
    'beckoned': 'called',
    'bile': 'vomit',
    'bustling': 'busy',
    'bustled': 'hurried',
    'cacophony': 'noise',
    'ceaseless': 'endless',
    'clandestine': 'secret',
    'cloying': 'sickening',
    'croaked': 'yelled',
    'clang': 'noise',
    'comforting': 'soothing',
    'contorted': 'twisted',
    'determined': 'resolute',
    'disquiet': 'unease',
    'disarray': 'a mess',
    'dilapidated': 'falling apart',
    'ceased': 'stopped',
    'crescendo': '',
    'din': 'noise',
    'departed': 'left',
    'echoes': 'reverberations',
    'echoed': 'reverberated',
    'echoing': 'bouncing',
    'enigma': 'mystery',
    'ever-present': '',
    'facade': 'front',
    'footfall': 'step',
    'footfalls': 'Footsteps',
    'foreboding': 'dread',
    'falter': 'hesitate',
    'faltered': 'hesitated',
    'façade': 'front',
    'foliage': 'leaves',
    'form': 'body',
    'fled': 'ran',
    'flank': 'side',
    'jolted': 'jumped',
    'gloom': 'darkness',
    'gorge': 'throat',
    'grotesque': 'ugly',
    'grotesquely': '',
    'inexorably': 'relentlessly',
    'hulking': 'massive',
    'halt': 'stop',
    'halted': 'stopped',
    'incredulously': 'amazingly',
    'idyllic': 'beautiful',
    'labyrinthine': 'complex',
    'looming': 'impending',
    'looms': 'emerges',
    'loathsome': '',
    'macabre': 'grim',
    'maw': 'jaws',
    'monotonous': 'boring',
    'murmured': 'whispered',
    'manacles': 'handcuffs',
    'malevolent': 'evil',
    'midst': 'middle of',
    'normalcy': 'normality',
    'oppressive': '',
    'palpable': 'tangible',
    'pang': 'sense',
    'pallid': 'pale',
    'pumping': 'pulsating',
    'jostled': 'bumped',
    'resolve': 'determination',
    'resolved': 'determined',
    'rythmic': '',
    'remain': 'stay',
    'regaling': 'entertaining',
    'regaled': 'entertained',
    'raucous': 'loud',
    'sanctuary': 'refuge',
    'scanned': 'searched',
    'sentinel': 'guard',
    'sentinels': 'guards',
    'shrill': 'piercing',
    'sinewy': 'muscular',
    'sinister': 'menacing',
    'solitary': 'lonely',
    'solitude': 'loneliness',
    'slumber': 'sleep',
    'spectral': 'ghostly',
    'stark': 'harsh',
    'stifling': 'suffocating',
    'steeled': 'braced',
    'sturdy': 'strong',
    'scanned': 'searched',
    'symphony': 'harmony',
    'tangible': 'real',
    'tapestry': 'fabric',
    'testament': 'proof',
    'threadbare': 'worn',
    'thrummed': 'vibrated',
    'tendrils': 'tentacles',
    'tomes': 'books',
    'tinge': 'trace',
    'tinged': 'colored',
    'trepidation': 'fear',
    'throng': 'crowd',
    'twitched': 'shook',
    'unwavering': 'steady',
    'undulated': 'waved',
    'unflappable': 'calm',
    'uneasy': 'nervous',
    'undergrowth': 'shrubbery',
    'wavered': 'hesitated',
    'whirled': 'spun',
    'vigil': 'watch',
    'vast': 'large',
    }

    phrase_bank = {
        'I frowned. ': '',
        ', frowning': '',
        'I frowned and ': 'I',
        '\n---': '',
        '---\n': '',
        'a grotesque': 'an ugly',
        'long shadows': 'shadows',
        ' the midst of': '',
        ', and all-too-real': '',
        'an ancient-looking': 'an old',
        # Add other phrases here
    }

    # First replace phrases (do these first to avoid word replacements breaking phrases)
    for old, new in phrase_bank.items():
        old = old.strip()
        if old in text:
            count = text.count(old)
            if count > 0:
                text = text.replace(old, new)

    # Then replace individual words
    words = text.split()
    for old, new in word_bank.items():
        old = old.strip()
        count = sum(1 for word in words if word.strip('.,!?";:').lower() == old.lower())
        if count > 0:
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)

    return text

def write_scene_transition(scene1: str, scene2: str) -> str:
    """Write a smooth transition paragraph between two scenes"""
    # Split scenes into paragraphs and clean empty lines
    scene1_paragraphs = [p for p in scene1.split('\n\n') if p.strip()]
    scene2_paragraphs = [p for p in scene2.split('\n\n') if p.strip()]
    
    # Get last/first 6 paragraphs (or all if less than 6)
    last_paragraphs = scene1_paragraphs[-6:] if len(scene1_paragraphs) > 6 else scene1_paragraphs
    first_paragraphs = scene2_paragraphs[:6] if len(scene2_paragraphs) > 6 else scene2_paragraphs

    first_scene_ending = '\n\n'.join(last_paragraphs)
    second_scene_beginning = '\n\n'.join(first_paragraphs)
    
    prompt = f"""
    Write a concise scene transition that smoothly transitions between these two scenes.
    The transition should connect the ending of the first scene to the beginning of the second scene.
    Focus on the passage of time and/or change in location that occurs between scenes.
    Only return the transition paragraph, no additional comments.

    First Scene Ending:
    {first_scene_ending}

    Second Scene Beginning:
    {second_scene_beginning}
    """

    try:
        response = or_client.chat.completions.create(
            model=settings.OR_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error writing transition: {e}")
        return ""

def main(username: str, channel_name: str):
    try:
        # Initialize both general and channel-specific settings
        settings.initialize_settings(username)
        settings.initialize_channel_settings(username, channel_name)
        
        # Initialize the clients with the loaded settings
        initialize_clients()
        
        # Generate story using channel-specific settings
        print("Generating story idea...")
        story_idea = story_ideas()
        print(story_idea)
        if story_idea is None:
            raise ValueError("Failed to generate story idea")
        print("Story idea generated successfully")
        
        print("Creating outline...")
        outline = create_outline(idea=story_idea, num=settings.NUM_SCENES)
        if outline is None:
            raise ValueError("Failed to create outline")
        print("Outline created successfully")
        
        print("Generating characters...")
        char = characters(outline)
        if char is None:
            raise ValueError("Failed to generate characters")
        print("Characters generated successfully")
        
        print("Writing story...")
        story, edited_scenes, original_scenes = write_story(outline, char)
        if not story or not edited_scenes:
            raise ValueError("Failed to write story")
        print("Story written successfully")

        # Save all versions of the story
        with open(f'{channel_name}_final_story.txt', 'w', encoding='utf-8') as file:
            file.write('\n\n\n\n'.join(edited_scenes))
        with open(f'{channel_name}_original_story.txt', 'w', encoding='utf-8') as file:
            if isinstance(original_scenes, list):
                file.write('\n\n\n\n'.join(original_scenes))
        #     else:
        #         file.write(original_scenes)
        print("All story versions saved to files")

        return story, edited_scenes, story_idea
        
    except Exception as e:
        print(f"Error in story_writer.main: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None, None, None
    

def main2(username: str, channel_name: str):
    # Initialize both general and channel-specific settings
    settings.initialize_settings(username)
    settings.initialize_channel_settings(username, channel_name)
        
    # Initialize the clients with the loaded settings
    initialize_clients()

    # Read the story from file
    try:
        with open(f'{channel_name}_rephrased_story.txt', 'r', encoding='utf-8') as file:
            story = file.read()
            # Split by 4 newlines to get scenes
            edited_scenes = story.split('\n\n\n\n')
            # edited_scenes = [callTune3(scene) for scene in edited_scenes]
            # Clean up any empty scenes
            edited_scenes = [scene.strip() for scene in edited_scenes if scene.strip()]
    except FileNotFoundError:
        print(f"Could not find {channel_name}_original_story.txt")
        return None, None, None
    except Exception as e:
        print(f"Error reading story file: {str(e)}")
        return None, None, None

    story = '\n\n'.join(edited_scenes)
    story_idea = story
    return story, edited_scenes, story_idea

def main3(username: str, channel_name: str):
    # Initialize both general and channel-specific settings
    settings.initialize_settings(username)
    settings.initialize_channel_settings(username, channel_name)
        
    # Initialize the clients with the loaded settings
    initialize_clients()

    # Read the story from file
    try:
        with open(f'{channel_name}_rephrased_story.txt', 'r', encoding='utf-8') as file:
            story = file.read()
            # Split by 4 newlines to get scenes
            edited_scenes = story.split('\n\n\n\n')
            # Clean up any empty scenes before processing
            scenes_to_process = [scene.strip() for scene in edited_scenes if scene.strip()]
            
            # Process each scene with callTune5 using progress bar
            processed_scenes = []
            with tqdm(total=len(scenes_to_process), desc="Processing scenes", unit="scene") as pbar:
                for scene in scenes_to_process:
                    processed_scene = callTune4(scene)
                    processed_scenes.append(processed_scene)
                    pbar.update(1)
            
            # Save the rewritten story
            with open(f'{channel_name}_rewritten_story.txt', 'w', encoding='utf-8') as outfile:
                outfile.write('\n\n\n\n'.join(processed_scenes))
                
    except FileNotFoundError:
        print(f"Could not find {channel_name}_rephrased_story.txt")
        return None, None, None
    except Exception as e:
        print(f"Error processing story: {str(e)}")
        return None, None, None

    story = '\n\n'.join(processed_scenes)
    story_idea = story
    return story, processed_scenes, story_idea


if __name__ == "__main__":
    # Example usage
    username = "229202"  # Replace with actual username
    channel_names = settings.get_channel_names(username)
    if channel_names:
        main(username, channel_names[0])
    else:
        print("No channels found for user")

