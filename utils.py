import requests
import constants
import os
from PIL import Image
from gradio_client import Client
import moviepy.editor as mp
from moviepy.video.VideoClip import ImageClip
from moviepy.editor import AudioFileClip
from structured_output_extractor import StructuredOutputExtractor
from pydantic import BaseModel, Field
from typing import List
import tempfile
import os


def clean_response(result):
    print("\n\nStarted Cleaning Response")
    """A temporary fix to the output of predict which returns output of openai-whisper-large-v3-turbo as string
    but it outputs: AutomaticSpeechRecognitionOutput(text=" sometimes life   <- like this the class name still remains
    in the response, ideally which should have started from "sometimes..." as in the given example  """
    # Use find() to get the position of the start and end of the text
    start_pos = result.find('text="') + len('text="')  # Start after 'text="'
    end_pos = result.find('", chunks=None')  # End before '", chunks=None'

    # Extract the text using slicing
    cleaned_result = result[start_pos:end_pos]
    print("Returning Cleaned Result: ", cleaned_result)
    return cleaned_result


def get_translation(text: str):
    print('\n\nTranslating text: ', text, type(text))
    # Input payload
    data = {"text_input": text}

    # Headers for authentication
    headers = {"Authorization": f"Bearer {constants.HF_TOKEN}"}

    try:
        # Make a GET request
        response = requests.post(constants.TRANSLATION_ENDPOINT, json=data, headers=headers)
        # Process response
        if response.status_code == 200:
            response_data = response.json()
            print("Returning Translation")
            return response_data.get("output", "No output found.")
        else:
            print("Some Error Occured During Translation Request")
            print(response)
            print(f"Error: {response.status_code}, {response.text}")
            return {"error_occured" : response.text}
    except Exception as e:
        print(f"An exception occurred: {e}")
        return {"error_occured" : e}
    
 
def segments_to_chunks(segments):
    chunks = []
    for segment in segments:
        chunks.append(segment.get("text"))
    return chunks
    

def get_image_prompts(text_input : List):
        # Example Pydantic model (e.g., Movie)
    class ImagePromptResponseSchema(BaseModel):
        image_prompts: List[str] = Field(
            description="List of detailed image prompts, Each Image Prompt Per Chunk"
        )

    extractor = StructuredOutputExtractor(response_schema=ImagePromptResponseSchema)
    chunks_count = len(text_input)
    chunks = "chunk: " + "\nchunk: ".join(text_input)
    prompt = f"""ROLE: You are a Highly Experienced Image Prompt Sythesizer (try to avoid explicit unethical prompt gracefully as much as possible) 
    TASK:  Generate {chunks_count} image prompts, Each per chunk\n\n {chunks}"""
    result = extractor.extract(prompt)
    return result.model_dump()   # returns dictionary version pydantic model
    
    



def generate_image(prompt, path='test_image.png'):
    try:
        # Initialize the Gradio Client with Hugging Face token
        client = Client(constants.IMAGE_GENERATION_SPACE_NAME, hf_token=constants.HF_TOKEN)

        # Make the API request
        result = client.predict(
            param_0=prompt,  # Text prompt for image generation
            api_name="/predict"
        )

        image = Image.open(result)
        image.save(path)

        # Return the result (which includes the URL or file path)
        return result

    except Exception as e:
        print(f"Error during image generation: {e}")
        return {"error": str(e)}
    
def generate_images(image_prompts, folder_name='test_folder'):
    folder_path = tmp_folder(folder_name)
    for index, prompt in enumerate(image_prompts):
        print(index, prompt)
        image_path = generate_image(prompt=prompt, path=f"{folder_path}/{index}.png")
        yield prompt, image_path
    


def tmp_folder(folder_name: str) -> str:
    # Use the current working directory or any other accessible path for temp folders
    base_tmp_path = os.path.join(os.getcwd(), "tmp_dir")  # Change this to any path you prefer
    
    # Ensure that the base temp folder exists
    if not os.path.exists(base_tmp_path):
        os.makedirs(base_tmp_path)
        print(f"Base temporary folder '{base_tmp_path}' created.")
    
    # Define the path for the specific temporary folder
    folder_path = os.path.join(base_tmp_path, folder_name)
    
    # Create the specific temporary folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"Temporary folder '{folder_name}' is ready at {folder_path}.")
    
    return folder_path



from moviepy.editor import *


import os
import tempfile
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips


def generate_video(audio_file, images, segments):
    try:
        # Save the uploaded audio file to a temporary location
        file_extension = os.path.splitext(audio_file.name)[1]
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"{file_extension}")
        temp_audio_path.write(audio_file.read())
        temp_audio_path.close()

        # Load the audio file using MoviePy
        audio = AudioFileClip(temp_audio_path.name)

        # Define YouTube-like dimensions (16:9 aspect ratio)
        frame_width = 1280
        frame_height = 720

        video_clips = []
        total_segments = len(segments)

        for i, current_segment in enumerate(segments):
            start_time = current_segment["start"]
            end_time = current_segment["end"]
            
            # Calculate the actual duration including any gap until the next segment
            if i < total_segments - 1:
                # If there's a next segment, extend until it starts
                next_segment = segments[i + 1]
                actual_end_time = next_segment["start"]
            else:
                # For the last segment, use its end time
                actual_end_time = end_time
            
            # Calculate total duration including any gap
            segment_duration = actual_end_time - start_time
            
            print(f"\nProcessing segment {i + 1}/{total_segments}:")
            print(f"  Start time: {start_time}s")
            print(f"  Base end time: {end_time}s")
            print(f"  Actual end time: {actual_end_time}s")
            print(f"  Total duration: {segment_duration}s")
            print(f"  Text: '{current_segment['text']}'")
            
            # Ensure the image index is within bounds
            image_path = images[min(i, len(images) - 1)]
            
            # Create an ImageClip for the current segment
            image_clip = ImageClip(image_path)
            
            # Resize and pad the image to fit a 16:9 aspect ratio
            image_clip = image_clip.resize(height=frame_height).on_color(
                size=(frame_width, frame_height),
                color=(0, 0, 0),  # Black background
                pos="center"      # Center the image
            )
            
            # Set the duration and start time for the clip
            image_clip = image_clip.set_duration(segment_duration)
            image_clip = image_clip.set_start(start_time)  # Set the start time explicitly
            
            video_clips.append(image_clip)

        # Concatenate all the image clips to form the video
        print("Concatenating video clips...")
        video = concatenate_videoclips(video_clips, method="compose")

        # Add the audio to the video
        video = video.set_audio(audio)

        # Save the video to a temporary file
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, "generated_video.mp4")
        print(f"Writing video file to {video_path}...")
        video.write_videofile(video_path, fps=30, codec="libx264", audio_codec="aac")

        # Clean up the temporary audio file
        os.remove(temp_audio_path.name)
        print("Temporary audio file removed.")

        return video_path

    except Exception as e:
        print(f"Error generating video: {e}")
        return None






# Example usage:
if __name__ == "__main__":
    result = generate_images(["a guy in jungle", "a waterfall","greenery"])


