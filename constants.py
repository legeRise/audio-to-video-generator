from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
AUDIO_CONVERTER_ENDPOINT="https://audio-converter-api-587c.onrender.com/convert/mp3"


TRANSLATION_ENDPOINT="https://habib926653-text-translator-agent-api.hf.space/generate"
PROMPT_GENERATION_ENDPOINT="https://habib926653-text-translator-agent-api.hf.space/get-image-prompts"
IMAGE_GENERATION_SPACE_NAME="habib926653/stabilityai-stable-diffusion-3.5-large-turbo"

# Supported formats
SUPPORTED_FORMATS = ["mp3", "wav", "ogg", "flac", "aac", "m4a"]