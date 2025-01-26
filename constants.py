from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)

SUMMARIZATION_ENDPOINT="https://habib926653-text-translator-agent-api.hf.space/generate"
IMAGE_GENERATION_SPACE_NAME="habib926653/stabilityai-stable-diffusion-3.5-large-turbo"

# Supported formats
SUPPORTED_FORMATS = ["mp3", "wav", "ogg", "flac", "aac", "m4a"]


