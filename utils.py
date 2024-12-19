
import requests
import constants
import os
from PIL import Image
from gradio_client import Client


def clean_response(result):
    """A temporary fix to the output of predict which returns output of openai-whisper-large-v3-turbo as string
    but it outputs: AutomaticSpeechRecognitionOutput(text=" sometimes life   <- like this the class name still remains
    in the response, ideally which should have started from "sometimes..." as in the given example  """
    # Use find() to get the position of the start and end of the text
    start_pos = result.find('text="') + len('text="')  # Start after 'text="'
    end_pos = result.find('", chunks=None')  # End before '", chunks=None'

    # Extract the text using slicing
    cleaned_result = result[start_pos:end_pos]

    return cleaned_result


def get_translation(text: str):
    # Input payload
    params = {"text": text}

    # Headers for authentication
    headers = {"Authorization": f"Bearer {constants.HF_TOKEN}"}

    try:
        # Make a GET request
        response = requests.get(constants.TRANSLATION_ENDPOINT, params=params, headers=headers)

        # Process response
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("output", "No output found.")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None
    


def get_image_prompts(text_input):
    headers = {
        "Authorization": f"Bearer {constants.HF_TOKEN}",  # Replace with your token
        "Content-Type": "application/json"  # Optional, ensures JSON payload
    }

    endpoint = f"{constants.PROMPT_GENERATION_ENDPOINT}"
    payload = {"text_input": text_input}
    
    try:
            # Send the POST request
        print("making post request for image prompts", endpoint)
        response = requests.post(endpoint, json=payload, headers=headers)
        
        # Raise an exception for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return {"error": str(e)}
    



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


def generate_video(image_folder, audio):
    return os.path.join(os.getcwd(), "test.mp4")


# Example usage:
if __name__ == "__main__":
    result = generate_images(["a guy in jungle", "a waterfall","greenery"])
    print(result,'is the result')
