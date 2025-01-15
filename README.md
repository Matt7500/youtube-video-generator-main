
# Story Narration Video Generator

This is a program I developed to automatically create horror story narration YouTube videos (it can work with other story genres as well).


This is made possible through the use of the following services:

- OpenAI
- OpenRouter
- ElevenLabs
- Replicate
- AWS
- MongoDB

## How it works

Story writing

- First, it generates a story idea using my GPT 4o fine-tune that was trained on the stories uploaded by other popular channels.

- Then it uses that idea and creates a plot outline for the story that is between 6 and 10 scenes(chapters) long in order to get the desired video length.

- After that it generates a list of characters for the story and information about them.

- It will write the story scene by scene and use information from the previous 2 scenes and the list of characters to write the scene beat it's been given to maintain continuity in the story.

- After the scene is written then it will be checked for any errors in the scene that conflict with the continuity of the previous scenes. Then it will check the re-written scene for any more errors before moving to the next one.

- Once the story is fully written it will then go through and rewrite the entire story using two GPT 4o fine-tunes, one for the dialoge and one for the narration. The purpose of these fine tunes is to make the writing sound more human and less like AI.

Title and Thumbnail Generation

- After the story is fully complete then it will generate the title and thumbnail for the video.

- To do this, I created a GPT 4o fine-tune that was trained on hundreds of youtube videos and their titles to get the style I wanted. It will take the original story idea and create a tile from that.

- Then, I use the Replicate API to create the background image automatically by having the AI create a prompt based on the story idea. After that, the title is imposed on the background image using Pillow.

Video Generation

- Then, the video generation starts by creating an AI voiceover using the ElevenLabs API. It breaks up the scenes into 5 paragaph segments in order to maintain consistency in the voice quality, then they are stitched back together.

- Once the audio is finished then the program will start (or create) and AWS EC2 instance where the video will be generated on for the compute power and the incredibly fast upload speed so the videos can be uploaded instantly from anywhere in the world, no matter my internet speed.

- Then it will run through the program and create the video based on the settings in the profile on the MongoDB database and automatically upload and schedule the video at the date specified in the database (this is updated +1 day after each video is created).