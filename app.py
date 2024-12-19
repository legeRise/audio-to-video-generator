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

# Function to convert the audio to MP3 using the external API
def convert_to_mp3(audio_file):
    if audio_file.name.endswith(".mp3"):
        return audio_file, False  # File is already MP3
    else:
        # Send to the external converter API
        url = constants.AUDIO_CONVERTER_ENDPOINT
        files = {"file": (audio_file.name, audio_file, "audio/mpeg")}

        with st.spinner("Converting audio to MP3... Please wait."):
            response = requests.post(url, files=files)

        if response.status_code == 200:
            # If conversion is successful, save and return the MP3 file
            converted_file = io.BytesIO(response.content)
            converted_file.name = "converted.mp3"
            return converted_file, True  # File was converted
        else:
            st.error("Conversion failed. Please try another format.")
            return None, None

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>AI Video Generator</h1>",
    unsafe_allow_html=True
)

# Upload audio file
audio_file = st.file_uploader("🔼 Upload your audio file:", type=constants.SUPPORTED_FORMATS)

if audio_file:
    # Reset states when a new file is uploaded
    if st.session_state.uploaded_file_name != audio_file.name:
        st.session_state.uploaded_file_name = audio_file.name
        st.session_state.converted_audio, st.session_state.was_converted = convert_to_mp3(audio_file)
        st.session_state.transcript = None
        st.session_state.translation = None
        st.session_state.generated_video = None  # Reset video generation state

    # Display uploaded file name
    st.info(f"Uploaded file: **{audio_file.name}**")

    if st.session_state.converted_audio:
        if not st.session_state.was_converted:
            st.success("🎧 The uploaded file is already in MP3 format.")
        else:
            st.success("✅ File successfully converted to MP3!")

        # Save the file temporarily if no transcript exists
        if st.session_state.transcript is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(st.session_state.converted_audio.read())
                tmp_file_path = tmp_file.name

            result = st.session_state.client.predict(
                param_0=handle_file(tmp_file_path),
                api_name="/predict"
            )
            st.session_state.transcript = clean_response(result)

            # Clean up temporary file
            os.remove(tmp_file_path)

        # Ensure translation is always generated after transcription
        if st.session_state.transcript and st.session_state.translation is None:
            with st.spinner("Generating translation..."):
                st.session_state.translation = get_translation(st.session_state.transcript)

        # Display and allow playback of the MP3 file
        st.audio(st.session_state.converted_audio, format="audio/mp3")

        # Toggle to show or hide the transcript
        toggle_transcript = st.checkbox("Show Transcript", value=st.session_state.transcript_visible)

        if toggle_transcript:
            st.session_state.transcript_visible = True
            st.write("### Transcription:")
            st.write(st.session_state.transcript)
        else:
            st.session_state.transcript_visible = False

        # Toggle to show or hide the translation
        toggle_translation = st.checkbox("Show Translation", value=st.session_state.translation_visible)

        if toggle_translation:
            st.session_state.translation_visible = True
            st.write("### Translation:")
            st.write(st.session_state.translation)
        else:
            st.session_state.translation_visible = False

        # Image prompts - generated once translation is available
        if st.session_state.translation:
            st.write("### Image Prompts")

            # Determine whether to use translation or transcription for image generation
            prompts = []
            if 'Already in English' in st.session_state.translation:
                st.info("Audio is Already in English. Using Transcription to generate Image Prompts")
                prompts = get_image_prompts(st.session_state.transcript)['image_prompts']
            else:
                prompts = get_image_prompts(st.session_state.translation)['image_prompts']

            # Display the prompts
            for i, prompt in enumerate(prompts):
                st.write(f"**Prompt {i+1}:** {prompt}")


            # Generate and display images using the generator
            for prompt, image_path in generate_images(prompts):
                st.image(image_path, caption=f"Prompt: {prompt}", use_column_width=True)
                st.write(f"Generated from: {prompt}")

            st.info("Video Generation Feature Currently Under Development")
                # # Generate the video based on the images and translation
                # st.write("### Generating Video...")
                # with st.spinner("Creating video..."):
                #     video_file = generate_video(images_folder, st.session_state.translation)
                #     if video_file:
                #         st.session_state.generated_video = video_file
                #         st.video(video_file)  # Display the video
                #     else:
                #         st.error("Failed to generate the video.")

else:
    # If no file is uploaded yet
    st.warning("Please upload an audio file to proceed.")
