import streamlit as st
import requests
import io
from gradio_client import Client, handle_file
import tempfile
import os
from utils import clean_response, get_translation, get_image_prompts, generate_images, generate_video
import constants  

# Initialize the client only once
if 'client' not in st.session_state:
    st.session_state.client = Client("habib926653/openai-whisper-large-v3-turbo", hf_token=constants.HF_TOKEN)

# Initialize state variables
if 'transcript_visible' not in st.session_state:
    st.session_state.transcript_visible = False
if 'translation_visible' not in st.session_state:
    st.session_state.translation_visible = False
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'converted_audio' not in st.session_state:
    st.session_state.converted_audio = None
if 'was_converted' not in st.session_state:
    st.session_state.was_converted = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'translation' not in st.session_state:
    st.session_state.translation = None
if 'generated_video' not in st.session_state:
    st.session_state.generated_video = None
if 'image_prompts' not in st.session_state:
    st.session_state.image_prompts = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = None

# Function to convert the audio to MP3 using the external API
def convert_to_mp3(audio_file):
    if audio_file.name.endswith(".mp3"):
        return audio_file, False  # File is already MP3
    else:
        # Send to the external converter API
        url = constants.AUDIO_CONVERTER_ENDPOINT
        files = {"file": (audio_file.name, audio_file, "audio/mp3")}

        with st.spinner("Converting audio to MP3... Please wait."):
            response = requests.post(url, files=files)

        if response.status_code == 200:
            # If conversion is successful, save and return the MP3 file
            converted_file = io.BytesIO(response.content)
            converted_file.name = "converted.mp3"
            st.success("✅ File successfully converted to MP3!")
            return converted_file, True  # File was converted
        else:
            st.error("❌ Conversion failed. Please try another format.")
            return None, None

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>AI Video Generator</h1>",
    unsafe_allow_html=True
)

# Upload audio file
audio_file = st.file_uploader("🔼 Upload your audio file:", type=constants.SUPPORTED_FORMATS)

if audio_file:
    # Reset states only when a new file is uploaded
    if st.session_state.uploaded_file_name != audio_file.name:
        st.session_state.uploaded_file_name = audio_file.name
        st.session_state.converted_audio, st.session_state.was_converted = convert_to_mp3(audio_file)
        st.session_state.transcript = None
        st.session_state.translation = None
        st.session_state.image_prompts = None
        st.session_state.generated_images = None  # Reset image generation state

    st.info(f"Uploaded file: **{audio_file.name}**")

    if st.session_state.converted_audio:
        if not st.session_state.was_converted:
            st.success("🎧 The uploaded file is already in MP3 format.")
        else:
            st.success("✅ File successfully converted to MP3!")

        # Transcription logic
        if st.session_state.transcript is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(st.session_state.converted_audio.read())
                tmp_file_path = tmp_file.name

            with st.spinner("Transcribing audio... Please wait."):
                result = st.session_state.client.predict(
                    param_0=handle_file(tmp_file_path),
                    api_name="/predict"
                )
                st.session_state.transcript = clean_response(result)
                os.remove(tmp_file_path)

        # Translation logic
        if st.session_state.transcript and st.session_state.translation is None:
            with st.spinner("Generating translation... Please wait."):
                st.session_state.translation = get_translation(st.session_state.transcript)

        st.audio(st.session_state.converted_audio, format="audio/mp3")

        # Toggle transcript visibility
        toggle_transcript = st.checkbox("Show Transcript", value=st.session_state.transcript_visible)
        st.session_state.transcript_visible = toggle_transcript

        if st.session_state.transcript_visible:
            st.write("### Transcription:")
            st.write(st.session_state.transcript)

        # Toggle translation visibility
        toggle_translation = st.checkbox("Show Translation", value=st.session_state.translation_visible)
        st.session_state.translation_visible = toggle_translation

        if st.session_state.translation_visible:
            st.write("### Translation:")
            st.write(st.session_state.translation)

        # Image generation logic
        if st.session_state.translation and st.session_state.image_prompts is None:
            with st.spinner("Generating image prompts... Please wait."):
                if 'Already in English' in st.session_state.translation:
                    st.info("Audio is Already in English. Using Transcription to generate Image Prompts")
                    st.session_state.image_prompts = get_image_prompts(st.session_state.transcript)['image_prompts']
                else:
                    st.session_state.image_prompts = get_image_prompts(st.session_state.translation)['image_prompts']


        # Ensure that generated_images is always a list
        if 'generated_images' not in st.session_state or st.session_state.generated_images is None:
            st.session_state.generated_images = []

        # Generate images only if they have not been generated already
        if st.session_state.image_prompts and not st.session_state.generated_images:
            with st.spinner("Generating images... Please wait."):
                for prompt, image_path in generate_images(st.session_state.image_prompts):
                    # Display each image as soon as it's generated
                    st.image(image_path, caption=f"{prompt}", use_container_width=True)
                    # Append the generated image to the session state
                    st.session_state.generated_images.append((prompt, image_path))

        # Display all previously generated images (including newly generated ones)
        else:
            for prompt, image_path in st.session_state.generated_images:
                # Display each image
                st.image(image_path, caption=f"{prompt}", use_container_width=True)

        st.info("Video Generation Feature Currently Under Development")
else:
    st.warning("Please upload an audio file to proceed.")

