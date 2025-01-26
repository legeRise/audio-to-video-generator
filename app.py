import streamlit as st
import os
import tempfile
import uuid
import logging
from utils import get_translation, get_image_prompts, segments_to_chunks, generate_images, generate_video
import constants  
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

client = Groq()

# Generate a unique session ID for each user
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session created with ID: {st.session_state.session_id}")

session_id = st.session_state.session_id

# Initialize state variables if not already set
state_variables = [
    'transcript_visible', 'translation_visible', 'uploaded_file_name', 
    'audio', 'was_converted', 'transcript', 'translation', 
    'generated_video', 'image_prompts', 'generated_images', 'video_generated'
]

for var in state_variables:
    if f'{var}_{session_id}' not in st.session_state:
        st.session_state[f'{var}_{session_id}'] = None
        logger.info(f"Initialized state variable: {var}_{session_id}")

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>AI Video Generator</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Leave a Like if it works for you! ❤️</p>", unsafe_allow_html=True)
st.info("**Video Generation Feature** - Functional But Can be Buggy")

# Encourage users to like the app
audio_option = st.radio("Choose audio input method:", ("Upload Audio File", "Record Audio"), horizontal=True)

if audio_option == "Upload Audio File":
    # Upload audio file
    audio_file = st.file_uploader("🔼 Upload your audio file:", type=constants.SUPPORTED_FORMATS)
else:
    audio_file = st.audio_input("🔊 Record Audio")

logger.debug(f"Audio option selected: {audio_option}")

if audio_file:
    logger.info(f"Audio file received: {audio_file.name}")

    # Reset states only when a new file is uploaded
    if st.session_state[f'uploaded_file_name_{session_id}'] != audio_file.name:
        st.session_state[f'uploaded_file_name_{session_id}'] = audio_file.name
        st.session_state[f'audio_{session_id}'] = audio_file
        st.session_state[f'transcript_{session_id}'] = None
        st.session_state[f'translation_{session_id}'] = None
        st.session_state[f'image_prompts_{session_id}'] = None
        st.session_state[f'generated_images_{session_id}'] = None  # Reset image generation state
        st.session_state[f'generated_video_{session_id}'] = None  # Reset generated video state
        st.session_state[f'video_generated_{session_id}'] = False  # Reset video generated flag
        logger.info("State variables reset due to new audio file upload.")

    # Read the uploaded file's bytes and send to Groq API for transcription
    file_bytes = audio_file.read()
    logger.debug("Audio file bytes read successfully.")

    # Create a transcription of the audio file using Groq API
    try:
        result = client.audio.transcriptions.create(
            file=(audio_file.name, file_bytes),  # Send the audio file content directly to the API
            model="whisper-large-v3-turbo",  # Model to use for transcription
            prompt="Take Note of Overall Context of the Audio",  # Optional context for better transcription accuracy
            response_format="verbose_json",  # Return detailed JSON response
            temperature=0.0,  # Control randomness in the transcription output
        )
        st.session_state[f'transcript_{session_id}'] = result.text
        st.session_state[f'segments_{session_id}'] = result.segments
        logger.info("Transcription created successfully.")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        st.error("An error occurred during transcription.")

    # Translation logic
    if st.session_state[f'transcript_{session_id}'] and st.session_state[f'translation_{session_id}'] is None:
        with st.spinner("Generating translation... Please wait."):
            st.session_state[f'translation_{session_id}'] = get_translation(st.session_state[f'transcript_{session_id}'])
            logger.info("Translation generated successfully.")

    st.audio(st.session_state[f'audio_{session_id}'], format=f"audio/{audio_file.type}")

    # Toggle transcript visibility
    toggle_transcript = st.checkbox("Show Transcript", value=st.session_state[f'transcript_visible_{session_id}'], key="toggle_transcript")
    st.session_state[f'transcript_visible_{session_id}'] = toggle_transcript

    if st.session_state[f'transcript_visible_{session_id}']:
        st.write("### Transcription:")
        st.write(st.session_state[f'transcript_{session_id}'])

    # Toggle translation visibility
    toggle_translation = st.checkbox("Show Translation", value=st.session_state[f'translation_visible_{session_id}'], key="toggle_translation")
    st.session_state[f'translation_visible_{session_id}'] = toggle_translation

    if st.session_state[f'translation_visible_{session_id}']:
        st.write("### Translation:")
        st.write(st.session_state[f'translation_{session_id}'])

    # Image generation logic
    if st.session_state[f'translation_{session_id}'] and st.session_state[f'image_prompts_{session_id}'] is None:
        with st.spinner("Generating image prompts... Please wait."):
            if 'Already in English' in st.session_state[f'translation_{session_id}']:
                st.info("Audio is Already in English. Using Transcription to generate Image Prompts")
                st.session_state[f'image_prompts_{session_id}'] = get_image_prompts(segments_to_chunks(st.session_state[f'segments_{session_id}']))['image_prompts']
            else:
                st.session_state[f'image_prompts_{session_id}'] = get_image_prompts(segments_to_chunks(st.session_state[f'segments_{session_id}']))['image_prompts']
            logger.info("Image prompts generated successfully.")

    # Ensure that generated_images is always a list
    if f'generated_images_{session_id}' not in st.session_state or st.session_state[f'generated_images_{session_id}'] is None:
        st.session_state[f'generated_images_{session_id}'] = []

    # Generate images only if they have not been generated already
    if st.session_state[f'image_prompts_{session_id}'] and not st.session_state[f'generated_images_{session_id}']:
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        total_images = len(st.session_state[f'image_prompts_{session_id}'])
        progress_placeholder.text(f"Generating images. Please be patient...")
        
        for idx, (prompt, image_path) in enumerate(generate_images(st.session_state[f'image_prompts_{session_id}'])):
            st.session_state[f'generated_images_{session_id}'].append((prompt, image_path))
            progress = (idx + 1) / total_images
            progress_bar.progress(progress)
            progress_placeholder.text(f"Generated image {idx + 1} of {total_images}: {prompt[:50]}...")
        
        progress_placeholder.text("✅ All images generated successfully!")
        progress_bar.empty()
        logger.info("All images generated successfully.")

    # Generate video when all images are generated
    if st.session_state[f'generated_images_{session_id}'] and st.session_state[f'audio_{session_id}'] and not st.session_state[f'video_generated_{session_id}']:
        with st.spinner("Generating video... Please wait."):
            # Create a temporary directory for the video
            temp_dir = tempfile.gettempdir()
            video_filename = f"generated_video_{session_id}.mp4"
            video_path = os.path.join(temp_dir, video_filename)

            # Map images to segments
            image_paths = [img[1] for img in st.session_state[f'generated_images_{session_id}']]
            generated_video_path = generate_video(
                audio_file=st.session_state[f'audio_{session_id}'], 
                images=image_paths, 
                segments=st.session_state[f'segments_{session_id}']
            )
            st.session_state[f'generated_video_{session_id}'] = generated_video_path
            st.session_state[f'video_generated_{session_id}'] = True  # Set the flag to True
            st.success("Video generated successfully!")
            logger.info("Video generated successfully.")

    # Display the generated video
    if st.session_state[f'generated_video_{session_id}']:
        st.video(st.session_state[f'generated_video_{session_id}'])
        
        # Add a download button for the generated video
        with open(st.session_state[f'generated_video_{session_id}'], "rb") as file:
            st.download_button(
                label="Download Video",
                data=file,
                file_name=f"generated_video_{session_id}.mp4",
                mime="video/mp4"
            )

else:
    st.warning("Please upload an audio file to proceed.")
    logger.warning("No audio file uploaded.")


